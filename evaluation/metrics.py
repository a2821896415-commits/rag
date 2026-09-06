"""
检索评估指标库（纯函数，无外部依赖，便于单测）。

约定：
- relevance: 长度 = 检索返回列表长度，第 i 个元素为该位置的相关性。
  - 二值模式: 1 表示命中相关文档, 0 表示不相关。
  - 分级模式: 连续值 [0,1]，用于 ndcg 等需要分级相关的指标。
- k: 仅考察前 k 个返回结果（1-indexed 位置）。
"""
import math
from typing import Sequence


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def recall_at_k(relevance: Sequence[int], k: int) -> float:
    """Recall@k = 前 k 命中相关数 / 全部相关数（这里以"前 k 是否含相关"累计）。"""
    rel = list(relevance[:k])
    hits = sum(1 for r in rel if r > 0)
    total_relevant = sum(1 for r in relevance if r > 0)
    return _safe_div(hits, total_relevant)


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    """Precision@k = 前 k 命中相关数 / k。"""
    rel = list(relevance[:k])
    if not rel:
        return 0.0
    hits = sum(1 for r in rel if r > 0)
    return _safe_div(hits, len(rel))


def hit_rate_at_k(relevance: Sequence[int], k: int) -> float:
    """HitRate@k = 前 k 是否至少命中一个相关文档（0/1）。"""
    return 1.0 if any(r > 0 for r in relevance[:k]) else 0.0


def mrr_at_k(relevance: Sequence[int], k: int) -> float:
    """MRR@k = 1 / 第一个相关文档的排名（超出 k 则为 0）。"""
    for i, r in enumerate(relevance[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(rels: Sequence[float]) -> float:
    """DCG = Σ (2^rel - 1) / log2(rank + 1)，rank 从 1 开始。"""
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(relevance: Sequence[float], k: int) -> float:
    """NDCG@k：分级相关下的归一化折损累计增益。

    relevance 可为二值(0/1)或连续[0,1]；ideal 取按相关度降序排列。
    """
    rels = list(relevance[:k])
    if not rels:
        return 0.0
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return _safe_div(_dcg(rels), idcg)


def f1_at_k(relevance: Sequence[int], k: int) -> float:
    """F1@k = 2PR / (P + R)。"""
    p = precision_at_k(relevance, k)
    r = recall_at_k(relevance, k)
    return _safe_div(2 * p * r, p + r)


def compute_retrieval_metrics(
    relevance: Sequence[float],
    k_list: Sequence[int],
    binary: bool = True,
) -> dict:
    """对一组（前 max_k 个）相关性打分，批量计算多个 k 下的检索指标。

    Args:
        relevance: 按返回顺序排列的相关性列表（二值或连续）。
        k_list: 需要计算的 k 列表，如 [1, 3, 5]。
        binary: True 表示按二值(>0 命中)计算 recall/precision/hit/mrr/f1；
                ndcg 始终使用原始连续值（分级）。
    Returns:
        dict: {k: {recall, precision, hit_rate, mrr, ndcg, f1}, ...}
    """
    rel_bin = [1 if r > 0 else 0 for r in relevance]
    out: dict = {}
    for k in k_list:
        out[k] = {
            "recall": round(recall_at_k(rel_bin, k), 4),
            "precision": round(precision_at_k(rel_bin, k), 4),
            "hit_rate": round(hit_rate_at_k(rel_bin, k), 4),
            "mrr": round(mrr_at_k(rel_bin, k), 4),
            "ndcg": round(ndcg_at_k(relevance, k), 4),
            "f1": round(f1_at_k(rel_bin, k), 4),
        }
    return out


def mean(values: Sequence[float]) -> float:
    values = [v for v in values if v is not None]
    return round(_safe_div(sum(values), len(values)), 4) if values else 0.0
