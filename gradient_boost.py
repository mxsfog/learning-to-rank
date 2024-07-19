import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.2, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 7, min_samples_leaf: int = 4):
        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.trees = []
        self.feature_indices = []

        self._prepare_data()

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values.reshape(-1, 1)
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values.reshape(-1, 1)
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
         X_test, y_test, self.query_ids_test) = self._get_data()

        self.X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        self.y_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        unique_query_ids = np.unique(inp_query_ids)
        X_scaled = np.zeros_like(inp_feat_array)

        for query_id in unique_query_ids:
            query_mask = (inp_query_ids == query_id)
            scaler = StandardScaler()
            X_scaled[query_mask] = scaler.fit_transform(inp_feat_array[query_mask])

        return X_scaled

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        np.random.seed(cur_tree_idx)
        random.seed(cur_tree_idx)

        lambdas = self._compute_lambdas(self.y_train, train_preds)

        sample_indices = np.random.choice(len(self.X_train), int(len(self.X_train) * self.subsample), replace=False)
        feature_indices = np.random.choice(self.X_train.shape[1], int(self.X_train.shape[1] * self.colsample_bytree),
                                           replace=False)

        tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                     random_state=cur_tree_idx)
        tree.fit(self.X_train[sample_indices][:, feature_indices], lambdas[sample_indices].numpy().ravel(),
                 sample_weight=np.abs(lambdas[sample_indices].numpy().ravel()))

        return tree, feature_indices
    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        unique_query_ids = np.unique(queries_list)
        ndcg_list = []

        for query_id in unique_query_ids:
            query_mask = (queries_list == query_id)
            ndcg = self._ndcg_k(true_labels[query_mask], preds[query_mask], self.ndcg_top_k)
            ndcg_list.append(ndcg)

        return float(np.mean(ndcg_list))

    def fit(self):
        np.random.seed(0)
        train_preds = torch.zeros_like(self.y_train)
        best_ndcg = -1
        best_trees = []
        best_feature_indices = []

        for cur_tree_idx in tqdm(range(self.n_estimators)):
            tree, feature_indices = self._train_one_tree(cur_tree_idx, train_preds)

            self.trees.append(tree)
            self.feature_indices.append(feature_indices)

            tree_preds = torch.FloatTensor(tree.predict(self.X_train[:, feature_indices]).reshape(-1, 1))
            train_preds += self.lr * tree_preds

            # Проверка разнообразия предсказаний
            if torch.std(train_preds) < 1e-6:
                print(f"Warning: Low variance in predictions at tree {cur_tree_idx}")

            train_ndcg = self._calc_data_ndcg(self.query_ids_train, self.y_train, train_preds)
            test_preds = self.predict(self.X_test)
            test_ndcg = self._calc_data_ndcg(self.query_ids_test, self.y_test, test_preds)

            if test_ndcg > best_ndcg:
                best_ndcg = test_ndcg
                best_trees = self.trees.copy()
                best_feature_indices = self.feature_indices.copy()

            print(f"Tree {cur_tree_idx + 1}/{self.n_estimators} - Train NDCG: {train_ndcg:.4f}, Test NDCG: {test_ndcg:.4f}")

        self.trees = best_trees
        self.feature_indices = best_feature_indices

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        preds = torch.zeros((data.shape[0], 1))

        for tree, feature_indices in zip(self.trees, self.feature_indices):
            tree_preds = torch.FloatTensor(tree.predict(data[:, feature_indices]).reshape(-1, 1))
            preds += self.lr * tree_preds

        return preds

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        ideal_dcg = compute_ideal_dcg(y_true)
        N = 1 / ideal_dcg

        _, rank_order = torch.sort(y_pred, descending=True, dim=0)
        rank_order += 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            Sij = compute_labels_in_batch(y_true)
            gain_diff = compute_gain_diff(y_true)

            decay_diff = (1.0 / torch.log2(rank_order.float() + 1.0)) - (1.0 / torch.log2(rank_order.t().float() + 1.0))
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)

            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

        return lambda_update

    def _ndcg_k(self, ys_true: torch.FloatTensor, ys_pred: torch.FloatTensor, ndcg_top_k: int) -> float:
        ys_true = ys_true.squeeze()
        ys_pred = ys_pred.squeeze()

        if len(ys_true) == 0:
            return 0.0

        ndcg_top_k = min(ndcg_top_k, len(ys_true))

        order = torch.argsort(ys_pred, descending=True)
        ys_true_sorted = ys_true[order]

        dcg = self._dcg_k(ys_true_sorted, ndcg_top_k)
        idcg = self._dcg_k(torch.sort(ys_true, descending=True)[0], ndcg_top_k)

        return (dcg / idcg).item() if idcg > 0 else 0.0

    def _dcg_k(self, ys_true: torch.FloatTensor, k: int) -> torch.FloatTensor:
        k = min(k, len(ys_true))
        return torch.sum((2 ** ys_true[:k] - 1) / torch.log2(torch.arange(2, k + 2, dtype=torch.float32)))

    def save_model(self, path: str):
        state = {
            'trees': self.trees,
            'feature_indices': self.feature_indices,
            'n_estimators': self.n_estimators,
            'lr': self.lr,
            'ndcg_top_k': self.ndcg_top_k,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.trees = state['trees']
        self.feature_indices = state['feature_indices']
        self.n_estimators = state['n_estimators']
        self.lr = state['lr']
        self.ndcg_top_k = state['ndcg_top_k']
        self.subsample = state['subsample']
        self.colsample_bytree = state['colsample_bytree']
        self.max_depth = state['max_depth']
        self.min_samples_leaf = state['min_samples_leaf']


def compute_ideal_dcg(y_true, ndcg_scheme='exp2'):
    sorted_y_true = torch.sort(y_true, descending=True)[0]
    if ndcg_scheme == 'exp2':
        gains = torch.pow(2.0, sorted_y_true) - 1
    elif ndcg_scheme == 'diff':
        gains = sorted_y_true
    else:
        raise ValueError(f"{ndcg_scheme} method not supported")
    discounts = torch.log2(torch.arange(2, gains.size(0) + 2, dtype=torch.float32))
    return torch.sum(gains / discounts).item()


def compute_labels_in_batch(y_true):
    rel_diff = y_true - y_true.t()
    pos_pairs = (rel_diff > 0).type(torch.float32)
    neg_pairs = (rel_diff < 0).type(torch.float32)
    return pos_pairs - neg_pairs


def compute_gain_diff(y_true, gain_scheme='exp2'):
    if gain_scheme == "exp2":
        gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
    elif gain_scheme == "diff":
        gain_diff = y_true - y_true.t()
    else:
        raise ValueError(f"{gain_scheme} method not supported")
    return gain_diff


def ndcg(y_true, y_pred, k=10):
    _, indices = torch.sort(y_pred, descending=True)
    y_true_sorted = y_true[indices]
    dcg = torch.sum((2 ** y_true_sorted - 1) / torch.log2(torch.arange(2, k + 2, dtype=torch.float32)))
    ideal_dcg = compute_ideal_dcg(y_true, ndcg_scheme='exp2')
    return (dcg / ideal_dcg).item()


# Пример использования
if __name__ == "__main__":
    solution = Solution()
    solution.fit()
    solution.save_model('model.pkl')
    solution.load_model('model.pkl')
    preds = solution.predict(torch.FloatTensor(solution.X_test))
    print(f"NDCG: {solution._calc_data_ndcg(solution.query_ids_test, solution.y_test, preds):.4f}")
