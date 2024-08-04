from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import precision_recall_curve
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes
from copy import deepcopy
from sklearn.metrics import auc as auc_fn


class EOL(AllInOne):
    def __init__(
        self,
        softmax_temperature: float,
        inference_steps: int,
        inference_lr: float,
        params2adapt: str,
        balancing_b: float,
        lambda_ce: float,
        lambda_marg: float,
        lambda_em: float,
    ):
        super().__init__()
        self.balancing_b = balancing_b
        self.lambda_ce = lambda_ce
        self.lambda_em = lambda_em
        self.lambda_marg = lambda_marg
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.softmax_temperature = softmax_temperature
        self.params2adapt = params2adapt

    def normalize_before_cosine(self, x):
        return F.normalize(x)

    def cosine(self, X, Y):
        return self.normalize_before_cosine(X) @ self.normalize_before_cosine(Y).T

    def clear(self):
        pass

    def get_logits(self, prototypes, query_features, log_scale=0, shift=0):
        logits = self.cosine(query_features, prototypes)  # [Nq, K]
        transformed_logits = torch.exp(log_scale) * logits + shift
        return self.softmax_temperature * transformed_logits

    def __call__(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self.iter_ = 0

        # Prepare data
        ns = support_features.size(0)
        nq = query_features.size(0)
        num_classes = support_labels.unique().size(0)
        Kshot = ns // num_classes
        all_features = torch.cat([support_features, query_features])
        targets_s = F.one_hot(support_labels)
        
        # Initialize weights
        with torch.no_grad():
            prototypes = compute_prototypes(support_features, support_labels)
        eta = torch.zeros(num_classes)
        delta = torch.zeros(num_classes)
        weights = torch.cat([
            torch.ones(num_classes) * Kshot / (1.0 - self.balancing_b),
            torch.ones(1) / self.balancing_b,
        ])
        
        # Prepare parameters for adaptation
        params_list = []
        if "prototypes" in self.params2adapt:
            prototypes.requires_grad_()
            params_list.append(prototypes)
        if "eta" in self.params2adapt:
            eta.requires_grad_()
            params_list.append(eta)
        if "delta" in self.params2adapt:
            delta.requires_grad_()
            params_list.append(delta)
        
        # Run adaptation
        optimizer = torch.optim.Adam(params_list, lr=self.inference_lr)

        # Adaptation loop
        for self.iter_ in range(self.inference_steps):
            # Compute logits and probabilities
            in_logits = self.get_logits(prototypes, all_features, log_scale=eta, shift=delta)
            in_probs = F.softmax(in_logits, dim=1) # P(y| x, S, Q, hat(y)=-1)
            
            # Compute inlier/outlier probabilities
            inlier_logit = - torch.logsumexp(in_logits, dim=1) + np.log(num_classes) - np.log(self.balancing_b)
            inlier_probs = torch.sigmoid(-inlier_logit)  # P(hat(y)=-1 | x, S, Q)
            
            # probs = in_probs * out_probs[:, :1] 
            probs = in_probs * inlier_probs[:, None] # P(y| x, S, Q)
            s_probs = probs[:ns]
            q_probs = probs[ns:]
            
            # Cross entropy on the support set
            ce_in = -(targets_s * torch.log(s_probs + 1e-12)).sum(1).mean(0)
            ce = self.lambda_ce * ce_in
            
            # Conditional entropy on the query set
            em_in = -(q_probs * torch.log(q_probs + 1e-12)).sum(1).mean(0)
            q_cond_ent = self.lambda_em * em_in / num_classes
            
            # Marginal entropy on the query set, weighted
            q_probs_mean = q_probs.mean(0) * weights[:num_classes]
            q_probs_mean_ood = (1 - q_probs.sum(1, keepdims=True)).mean(0) * weights[num_classes:]
            marg_in = (q_probs_mean * torch.log(q_probs_mean + 1e-12)).mean(0)
            marg_ood = (q_probs_mean_ood * torch.log(q_probs_mean_ood + 1e-12)).mean(0)
            q_ent = self.lambda_marg * (marg_in + marg_ood)
            
            # Compute Loss
            loss = ce + q_cond_ent + q_ent

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute outlier scores and probabilities using a refined estimation of the prototypes
        with torch.no_grad():
            # Compute logits and outlier scores
            in_logits = self.get_logits(prototypes, all_features, log_scale=eta, shift=delta)
            inlier_logit = - torch.logsumexp(in_logits, dim=1) + np.log(num_classes) - np.log(self.balancing_b)
            inlier_probs = torch.sigmoid(-inlier_logit)  # P(hat(y) | x, S, Q)
            outlier_scores = 1 - inlier_probs[ns:]
            
            # Create new prototypes based on the outlier-score prediction, do not update the outlier-scores again
            probs = in_logits.softmax(1) * inlier_probs[:, None]
            refined_prototypes = (
                probs.unsqueeze(2) * all_features.unsqueeze(1) / 
                    probs.sum(0, keepdim=True).unsqueeze(2)
            ).sum(0)
            in_logits = self.get_logits(refined_prototypes, all_features, log_scale=eta, shift=delta)
            probs = in_logits.softmax(1) * inlier_probs[:, None]
            s_probs = probs[:ns]
            q_probs = probs[ns:]
        
        # 
        return (
            s_probs.detach(),
            q_probs.detach(),
            outlier_scores.detach(),
        )
