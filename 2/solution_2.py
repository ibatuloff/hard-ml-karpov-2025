from functools import total_ordering
from math import log2
import torch
from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    # допишите ваш код здесь
    order = ys_true.argsort(descending=True)
    combs = torch.combinations(order, 2)
    num_inversions = ((ys_true[combs[:, 0]] != ys_true[combs[:, 1]]) & (ys_pred[combs[:, 0]] < ys_pred[combs[:, 1]])).sum().item() # now wtf is that. it works but... wtf
    return num_inversions



def compute_gain(y_value: float, gain_scheme: str) -> float:
    # допишите ваш код здесь
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2**y_value - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    # допишите ваш код здесь
    order = ys_pred.argsort(descending=True)
    inds = torch.arange(len(order), dtype=torch.float64) + 1
    result = 0
    for value, ind in zip(ys_true[order], inds):
        result += compute_gain(value, gain_scheme=gain_scheme) / torch.log2(ind + 1)
    return result



def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    # допишите ваш код здесь
    dcg_ = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    return dcg_ / ideal_dcg


def precision_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # допишите ваш код здесь
    total_relevant = ys_true.sum()
    if total_relevant == 0:
        return -1

    order = ys_pred.argsort(descending=True)[:k]
    n_retrieved = len(order)
    n_relevant = ys_true[order].sum().item()

    if n_retrieved > total_relevant:
        return n_relevant / total_relevant
    else:
        return n_relevant / n_retrieved


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    order = ys_pred.argsort(descending=True)
    true_label_pos = ys_true[order].argsort(descending=True)[0]
    return 1 / (true_label_pos + 1)



def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    # допишите ваш код здесь
    order = ys_pred.argsort(descending=True)

    p_rel = ys_true[order]
    p_look_prev = 1
    p_found = p_look_prev * p_rel[0]
    for i in range(1, len(order)):
        p_look = p_look_prev * (1 - p_rel[i-1]) * (1 - p_break)
        p_found += p_look * p_rel[i]
        p_look_prev = p_look

    return p_found




def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    if ys_true.sum() == 0: # if ys_true does not contain relevant document labels (no ones)
        return -1
    ap = 0
    recall_at_prev_k = 0 # for correct computing the difference between recall@k and recall@[k-1]
    order = ys_pred.argsort(descending=True) # for sorting the ys_true tensor
    for k in range(1, ys_true.size(0)+1):
        precision_at_k = ys_true[order][:k].sum() / k
        recall_at_k = ys_true[order][:k].sum() / ys_true.sum()
        ap += (recall_at_k - recall_at_prev_k) * precision_at_k
        recall_at_prev_k = recall_at_k
    return ap

