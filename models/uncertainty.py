"""
Uncertainty module for link anomaly detection in CTDGs.

`UncertaintyLinkPredictor` is a drop-in replacement for `ContextualLinkPredictor`: its `forward` returns a logit
whose sigmoid is the predictive probability, so the training loop does not change. It models three sources of
uncertainty about whether an edge e = (u, v, t, m) is benign:

    * aleatoric -- a heteroscedastic head predicts the variance of the logit. During training the likelihood is
      integrated over the logit noise with Monte Carlo samples (Kendall & Gal, 2017). This lets the model attenuate
      the loss on ambiguous training negatives, e.g., perturbed edges that are in fact plausible benign edges.
    * epistemic -- MC dropout in the prediction head (Gal & Ghahramani, 2016). The temporal backbone is evaluated
      only once per edge, so memory-based backbones are updated exactly as without the module.
    * support -- evidence-based uncertainty from subjective logic (Josang, 2016; Sensoy et al., 2018):
      u = W / (W + n), where n is the number of past observations of the node pair (u, v) and W is the prior weight.

At evaluation time an edge is scored with a lower confidence bound on its probability of being benign:

    score(e) = p(e) - sum_i lambda_i * u_i(e) / s_i,

where s_i is the standard deviation of u_i on the validation set. The weights lambda_i >= 0 are selected on the
validation set (the same data the original pipeline uses for model selection) and stored as buffers, so they are
saved and restored together with the model checkpoint. The test set is never used for fitting.
"""

import itertools
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.tgb_evaluate_linkanomdet import query_pred_edge_batch
from utils.DataLoader import Data
from utils.utils import NeighborSampler

UNCERTAINTY_COMPONENTS = ("aleatoric", "epistemic", "support")


class UncertaintyLinkPredictor(nn.Module):
    def __init__(
        self,
        input_dim_node1: int,
        input_dim_node2: int,
        input_dim_context: int,
        hidden_dim: int,
        output_dim: int = 1,
        components: Sequence[str] = UNCERTAINTY_COMPONENTS,
        mc_dropout: float = 0.1,
        num_mc_samples: int = 20,
        num_noise_samples: int = 10,
        init_logit_std: float = 0.1,
        support_prior_weight: float = 2.0,
        lambda_grid: Sequence[float] = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
        seed: int = 0,
    ):
        """
        Predicts a link based on two node embeddings and context, together with its uncertainty.
        :param input_dim_node1: int, dimension of the first node embedding
        :param input_dim_node2: int, dimension of the second node embedding
        :param input_dim_context: int, dimension of context
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output (only 1 is supported)
        :param components: sequence of str, uncertainty components to use, subset of UNCERTAINTY_COMPONENTS
        :param mc_dropout: float, dropout rate in the head (used in training and for MC dropout at evaluation)
        :param num_mc_samples: int, number of MC dropout samples at evaluation
        :param num_noise_samples: int, number of logit noise samples for the heteroscedastic likelihood in training
        :param init_logit_std: float, initial standard deviation of the logit noise
        :param support_prior_weight: float, prior weight W of the evidence-based support uncertainty
        :param lambda_grid: sequence of float, candidate weights of each uncertainty component
        :param seed: int, seed of the generator used for logit noise and dropout masks
        """
        super().__init__()
        assert output_dim == 1, "UncertaintyLinkPredictor supports only output_dim=1!"
        unknown_components = set(components) - set(UNCERTAINTY_COMPONENTS)
        if unknown_components:
            raise ValueError(
                f"Unknown uncertainty components {unknown_components}, choose from {UNCERTAINTY_COMPONENTS}!"
            )
        if "epistemic" in components and mc_dropout <= 0.0:
            raise ValueError("Epistemic uncertainty requires mc_dropout > 0!")

        self.components = [c for c in UNCERTAINTY_COMPONENTS if c in components]
        self.use_aleatoric = "aleatoric" in self.components
        self.use_epistemic = "epistemic" in self.components
        self.use_support = "support" in self.components
        self.mc_dropout = mc_dropout
        self.num_mc_samples = num_mc_samples
        self.num_noise_samples = num_noise_samples
        self.support_prior_weight = support_prior_weight
        self.lambda_grid = list(lambda_grid)
        self.seed = seed
        self._generator = None

        # Same layers, in the same order, as ContextualLinkPredictor, so both heads get identical initial weights.
        self.fc_down = nn.Linear(input_dim_context, input_dim_node1)
        self.fc1 = nn.Linear(
            input_dim_node1 + input_dim_node2 + input_dim_node1, hidden_dim
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

        if self.use_aleatoric:
            # Log-variance head, initialized deterministically (no draws from the global RNG) to a small constant
            # noise level, so that training starts from the same function as the original head.
            self.logvar_weight = nn.Parameter(torch.zeros(output_dim, hidden_dim))
            self.logvar_bias = nn.Parameter(
                torch.full((output_dim,), 2.0 * math.log(init_logit_std))
            )

        # Weights and scales of the uncertainty components, fitted on the validation set.
        self.register_buffer(
            "unc_weights", torch.zeros(len(UNCERTAINTY_COMPONENTS))
        )
        self.register_buffer("unc_scales", torch.ones(len(UNCERTAINTY_COMPONENTS)))

    def _get_generator(self, device: torch.device) -> torch.Generator:
        # A dedicated generator keeps the global RNG (used, e.g., for negative sampling) identical to the baseline.
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        return self._generator

    def _hidden(
        self,
        node_embedding1: torch.Tensor,
        node_embedding2: torch.Tensor,
        context: torch.Tensor,
    ):
        # Tensor, shape (*, input_dim_node1)
        projected_context = self.fc_down(context)
        # Tensor, shape (*, input_dim_node1 + input_dim_node2 + input_dim_node1)
        x = torch.cat([node_embedding1, node_embedding2, projected_context], dim=1)
        # Tensor, shape (*, hidden_dim)
        return self.act(self.fc1(x))

    def _dropout(self, h: torch.Tensor):
        if self.mc_dropout <= 0.0:
            return h
        keep = (
            torch.rand(h.shape, generator=self._get_generator(h.device), device=h.device)
            >= self.mc_dropout
        )
        return h * keep / (1.0 - self.mc_dropout)

    def _heads(self, h: torch.Tensor):
        # Mean and log-variance of the logit, shape (*, output_dim)
        mu = self.fc2(h)
        logvar = None
        if self.use_aleatoric:
            logvar = F.linear(h, self.logvar_weight, self.logvar_bias).clamp(-10.0, 5.0)
        return mu, logvar

    @staticmethod
    def _marginal_probability(mu: torch.Tensor, logvar: Optional[torch.Tensor]):
        # Probit approximation of E[sigmoid(mu + sigma * eps)], eps ~ N(0, 1).
        if logvar is None:
            return torch.sigmoid(mu)
        return torch.sigmoid(mu / torch.sqrt(1.0 + math.pi / 8.0 * logvar.exp()))

    def forward(
        self,
        node_embedding1: torch.Tensor,
        node_embedding2: torch.Tensor,
        context: torch.Tensor,
    ):
        """
        Predicts a link based on two node embeddings and context.
        Returns a logit whose sigmoid is the predictive probability, like ContextualLinkPredictor.
        :param node_embedding1: Tensor, shape (*, input_dim_node1)
        :param node_embedding2: Tensor, shape (*, input_dim_node2)
        :param context: Tensor, shape (*, input_dim_context)
        :return: Tensor, shape (*, output_dim)
        """
        h = self._hidden(node_embedding1, node_embedding2, context)
        if self.training:
            mu, logvar = self._heads(self._dropout(h))
            if logvar is None:
                return mu
            # Monte Carlo integration of the likelihood over the logit noise.
            eps = torch.randn(
                (self.num_noise_samples,) + mu.shape,
                generator=self._get_generator(mu.device),
                device=mu.device,
            )
            probability = torch.sigmoid(mu + (0.5 * logvar).exp() * eps).mean(dim=0)
        else:
            mu, logvar = self._heads(h)
            if logvar is None:
                return mu
            probability = self._marginal_probability(mu, logvar)
        return torch.logit(probability, eps=1e-6)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        node_embedding1: torch.Tensor,
        node_embedding2: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predicts the probability of a link together with its model-based uncertainties.
        :return: dictionary of Tensors with shape (*, ):
            'prob_deterministic' -- probability without MC dropout,
            'prob' -- predictive probability (MC dropout average if epistemic uncertainty is used),
            'aleatoric' -- standard deviation of the logit (if used),
            'epistemic' -- standard deviation of the probability across MC dropout samples (if used).
        """
        h = self._hidden(node_embedding1, node_embedding2, context)
        mu, logvar = self._heads(h)
        outputs = {
            "prob_deterministic": self._marginal_probability(mu, logvar).squeeze(dim=-1)
        }
        if self.use_aleatoric:
            outputs["aleatoric"] = (0.5 * logvar).exp().squeeze(dim=-1)
        if self.use_epistemic:
            # Tensor, shape (num_mc_samples, *)
            mc_probabilities = torch.stack(
                [
                    self._marginal_probability(*self._heads(self._dropout(h))).squeeze(dim=-1)
                    for _ in range(self.num_mc_samples)
                ]
            )
            outputs["prob"] = mc_probabilities.mean(dim=0)
            outputs["epistemic"] = mc_probabilities.std(dim=0)
        else:
            outputs["prob"] = outputs["prob_deterministic"]
        return outputs

    def combine(self, prob: np.ndarray, uncertainties: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Uncertainty-aware score (lower confidence bound on the probability of being benign).
        :param prob: ndarray, shape (num_edges, ), predictive probabilities
        :param uncertainties: dictionary of ndarrays with shape (num_edges, ), one for each used component
        :return: ndarray, shape (num_edges, ), scores where low values indicate anomalies
        """
        score = prob.astype(np.float64)
        for idx, name in enumerate(UNCERTAINTY_COMPONENTS):
            if name in self.components:
                weight = self.unc_weights[idx].item()
                scale = self.unc_scales[idx].item()
                score = score - weight * uncertainties[name] / scale
        return score

    def fit_uncertainty_weights(
        self,
        prob: np.ndarray,
        uncertainties: Dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> None:
        """
        Select the weights of the uncertainty components by grid search on AUC (labels: 1 benign, 0 anomalous).
        The grid contains all-zero weights, so the selected score is never worse than `prob` on the fitting data.
        Among equally good weights, the ones with the smallest sum are kept.
        """
        if len(np.unique(labels)) < 2:
            return

        scales = {
            name: max(float(np.std(uncertainties[name])), 1e-8) for name in self.components
        }
        normalized = {name: uncertainties[name] / scales[name] for name in self.components}

        best_auc = roc_auc_score(y_true=labels, y_score=prob)
        best_weights = [0.0] * len(self.components)
        for weights in sorted(
            itertools.product(self.lambda_grid, repeat=len(self.components)), key=sum
        ):
            if sum(weights) == 0.0:
                continue
            score = prob - sum(w * normalized[n] for w, n in zip(weights, self.components))
            auc = roc_auc_score(y_true=labels, y_score=score)
            if auc > best_auc:
                best_auc, best_weights = auc, list(weights)

        for weight, name in zip(best_weights, self.components):
            idx = UNCERTAINTY_COMPONENTS.index(name)
            self.unc_weights[idx] = weight
            self.unc_scales[idx] = scales[name]


def compute_support_uncertainty(
    neighbor_sampler: NeighborSampler,
    src_node_ids: np.ndarray,
    dst_node_ids: np.ndarray,
    node_interact_times: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    """
    Evidence-based uncertainty u = W / (W + n), where n is the number of interactions between the source and
    the destination node strictly before the edge's timestamp.
    :return: ndarray, shape (batch_size, )
    """
    num_observations = np.zeros(len(src_node_ids), dtype=np.float64)
    for idx, (src_node_id, dst_node_id, node_interact_time) in enumerate(
        zip(src_node_ids, dst_node_ids, node_interact_times)
    ):
        neighbor_ids, _, _, _ = neighbor_sampler.find_neighbors_before(
            node_id=src_node_id, interact_time=node_interact_time
        )
        num_observations[idx] = np.count_nonzero(neighbor_ids == dst_node_id)
    return prior_weight / (prior_weight + num_observations)


def eval_linkanomdet_TGB_with_uncertainty(
    model_name: str,
    model: nn.Module,
    neighbor_sampler: NeighborSampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_data: Data,
    evaluator,
    metrics: Optional[List[str]] = ["auc"],
    num_neighbors: Optional[int] = 20,
    time_gap: Optional[int] = 2000,
    show_progress_bar: Optional[bool] = False,
    fit_uncertainty_weights: Optional[bool] = False,
) -> Dict[str, float]:
    """
    Same as `eval_linkanomdet_TGB`, but scores edges with the uncertainty-aware score of `UncertaintyLinkPredictor`.
    Args:
        fit_uncertainty_weights (Optional[bool], optional): Whether to fit the weights of the uncertainty components
            on this data. Should be True only for the validation set. Defaults to False.

    Returns:
        Dict[str, float]: Dictonary with results. Besides the metrics of the uncertainty-aware score, it contains
            the metrics of the deterministic probability ('<metric>_mean_only'), the AUC of each uncertainty
            component alone ('auc_<component>_only'), and the selected weights ('lambda_<component>').
    """
    if model_name in ["DyRep", "TGAT", "TGN", "CAWN", "TCL", "GraphMixer", "DyGFormer"]:
        # Evaluation phase uses all the graph information.
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()
    predictor = model[1]

    outputs, y_labels = defaultdict(list), []
    with torch.no_grad():
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader,
            ncols=120,
            disable=(not show_progress_bar),
        )
        for batch_idx, evaluate_data_indices in enumerate(
            evaluate_idx_data_loader_tqdm
        ):
            (
                batch_src_node_ids,
                batch_dst_node_ids,
                batch_node_interact_times,
                batch_edge_ids,
                batch_labels,
            ) = (
                evaluate_data.src_node_ids[evaluate_data_indices],
                evaluate_data.dst_node_ids[evaluate_data_indices],
                evaluate_data.node_interact_times[evaluate_data_indices],
                evaluate_data.edge_ids[evaluate_data_indices],
                evaluate_data.labels[evaluate_data_indices],
            )
            batch_msg_feat = model[0].edge_raw_features[batch_edge_ids]

            # As in `eval_linkanomdet_TGB`, node memories are updated according to all edges.
            (
                batch_src_node_embeddings,
                batch_dst_node_embeddings,
            ) = query_pred_edge_batch(
                model_name=model_name,
                model=model,
                src_node_ids=batch_src_node_ids,
                dst_node_ids=batch_dst_node_ids,
                node_interact_times=batch_node_interact_times,
                edge_ids=batch_edge_ids,
                edges_are_positive=True,
                num_neighbors=num_neighbors,
                time_gap=time_gap,
            )

            batch_outputs = predictor.predict_with_uncertainty(
                node_embedding1=batch_src_node_embeddings,
                node_embedding2=batch_dst_node_embeddings,
                context=batch_msg_feat,
            )
            for key, value in batch_outputs.items():
                outputs[key].append(value.cpu().numpy())
            if predictor.use_support:
                outputs["support"].append(
                    compute_support_uncertainty(
                        neighbor_sampler=neighbor_sampler,
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        prior_weight=predictor.support_prior_weight,
                    )
                )
            y_labels.append(batch_labels)

    outputs = {key: np.concatenate(value) for key, value in outputs.items()}
    labels = np.concatenate(y_labels)

    if fit_uncertainty_weights:
        predictor.fit_uncertainty_weights(
            prob=outputs["prob"], uncertainties=outputs, labels=labels
        )
    scores = predictor.combine(prob=outputs["prob"], uncertainties=outputs)

    def evaluate(y_pred: np.ndarray) -> Dict[str, float]:
        return evaluator.eval(
            {"y_pred": y_pred, "y_label": labels, "eval_metric": metrics}
        )

    perf_metrics = evaluate(scores)
    for metric_name, metric_score in evaluate(outputs["prob_deterministic"]).items():
        perf_metrics[f"{metric_name}_mean_only"] = metric_score
    for name in predictor.components:
        idx = UNCERTAINTY_COMPONENTS.index(name)
        # Anomalies (label 0) should be uncertain, hence the uncertainty is negated.
        perf_metrics[f"auc_{name}_only"] = roc_auc_score(
            y_true=labels, y_score=-outputs[name]
        )
        perf_metrics[f"lambda_{name}"] = predictor.unc_weights[idx].item()
    return perf_metrics
