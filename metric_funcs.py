import torch


def num_swapped_pairs(ys_true, ys_pred):
    # Преобразуем входные данные в тензоры, если они таковыми не являются
    ys_true = torch.tensor(ys_true)
    ys_pred = torch.tensor(ys_pred)

    n = len(ys_true)
    num_pairs = 0

    # Проходим по всем парам (i, j) таких, что i < j
    for i in range(n):
        for j in range(i + 1, n):
            # Проверяем, является ли пара неправильно упорядоченной
            if (ys_true[i] > ys_true[j] and ys_pred[i] <= ys_pred[j]) or (ys_true[i] < ys_true[j] and ys_pred[i] >= ys_pred[j]):
                num_pairs += 1

    return num_pairs

def compute_gain(relevance, gain_sheme='exp2'):
    if gain_sheme == 'const':
        return relevance
    elif gain_sheme == 'exp2':
        return 2 ** relevance - 1
    else:
        raise ValueError('Incorrect gain scheme. Use "const" or "exp2".')

def dcg(relevance, gain_sheme='exp2'):
    relevances = torch.tensor(relevance, dtype=torch.float32)
    gains = torch.tensor([compute_gain(gain, gain_sheme=gain_sheme) for gain in relevances])
    discounts = torch.log2(torch.arange(2, gains.size(0) + 2, dtype=torch.float32))
    return torch.sum(gains / discounts).item()


def ndcg(ys_true, ys_pred, k):
    ys_true = torch.tensor(ys_true)
    ys_pred = torch.tensor(ys_pred)
    order = torch.argsort(ys_pred, descending=True)
    ys_true = ys_true[order]

    # Индексы для логарифмической шкалы
    idx = torch.arange(1, k + 1).float()

    # Обработаем случай, когда ys_true меньше чем k
    min_k = min(k, len(ys_true))

    # Вычисляем DCG и IDCG
    DCG = torch.sum((2 ** ys_true[:min_k] - 1) / torch.log2(idx[:min_k] + 1))
    sorted_ys_true = torch.sort(ys_true, descending=True).values
    IDCG = torch.sum((2 ** sorted_ys_true[:min_k] - 1) / torch.log2(idx[:min_k] + 1))

    return DCG / IDCG if IDCG > 0 else 0.0


def compute_ideal_dcg(y_true):
    _, rank_order = torch.sort(y_true, descending=True, dim=0)
    rank_order += 1
    gain_diff = torch.pow(2.0, y_true[rank_order - 1]) - 1
    decay = 1.0 / torch.log2(rank_order + 1.0)
    ideal_dcg = torch.sum(gain_diff * decay)
    return ideal_dcg


def precision_at_k(ys_true, ys_pred, k):
    ys_true = torch.tensor(ys_true)
    ys_pred = torch.tensor(ys_pred)

    if k > len(ys_true):
        k = len(ys_true)

    if torch.sum(ys_true) <= 0:
        return -1.0

    _, sorted_idxs = torch.sort(ys_pred, descending=True)

    top_k_idxs = sorted_idxs[:k]

    relevant_items_in_top_k = torch.sum(ys_true[top_k_idxs])
    precision = relevant_items_in_top_k.item() / k

    return precision

def reciprocal_rank(ys_true, ys_pred):
    ys_true = torch.tensor(ys_true)
    ys_pred = torch.tensor(ys_pred)

    _, sorted_idxs = torch.sort(ys_pred, descending=True)

    for rank, idx in enumerate(sorted_idxs, start=1):
        if ys_true[idx] == 1:
            return 1.0/rank

    return 0.0

def p_found(ys_true, p_break):
    ys_true = torch.tensor(ys_true, dtype=torch.float32)

    p_found_value = 0.0
    p_look = 1.0

    for relevance in ys_true:
        p_found_value += p_look * relevance
        p_look *= (1 - relevance) * (1 - p_break)

    return p_found_value

def avg_precision(ys_true, ys_pred):
    if torch.sum(ys_true) == 0:
        return -1.0

    sorted_idxs = torch.argsort(ys_pred, descending=True)
    ys_true_sorted = ys_true[sorted_idxs]
    ys_pred_sorted = ys_pred[sorted_idxs]

    tp_cumsum = torch.cumsum(ys_true_sorted, dim=0)
    precision = tp_cumsum / torch.arange(1, len(ys_true) + 1, dtype=torch.float32)

    ap = torch.sum((ys_pred_sorted * precision)) / torch.sum(ys_true)
    return ap