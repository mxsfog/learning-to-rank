import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from metric_funcs import ndcg
from catboost.datasets import msrank_10k


def listnet_kl_loss(y_i, z_i):
    """
    y_i: (n_i, 1) GT
    z_i: (n_i, 1) preds
    """
    y_i = torch.tensor(y_i, requires_grad=True)
    z_i = torch.tensor(z_i, requires_grad=True)

    P_y_i = torch.softmax(y_i, dim=0)
    P_z_i = torch.softmax(z_i, dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))


class ListNet(nn.Module):
    def __init__(self, num_input_features, hidden_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input):
        logits = self.model(input)
        return logits


class Solution():
    def __init__(self, n_epochs, listnet_hidden_dim, lr, ndcg_top_k, root, test_root):
        self.n_epochs = n_epochs
        self.listnet_hidden_dim = listnet_hidden_dim
        self.lr = lr
        self.ndcg_top_k = ndcg_top_k
        self.model = None
        self.train_data = root
        self.test_data = test_root

    def _get_data(self):
        self.X_train = self.train_data.iloc[:, 2:138].values
        self.y_train = self.train_data.iloc[:, 0].values
        self.query_ids_train = self.train_data.iloc[:, 1].values

        self.X_test = self.test_data.iloc[:, 2:138].values
        self.y_test = self.test_data.iloc[:, 0].values
        self.query_ids_test = self.test_data.iloc[:, 1].values

    def _scale_features_in_query_groups(self, X, query_ids):
        scaler = StandardScaler()
        unique_query_ids = np.unique(query_ids)
        X_scaled = np.zeros_like(X)
        for qid in unique_query_ids:
            indices = np.where(query_ids == qid)
            X_scaled[indices] = scaler.fit_transform(X[indices])
        return X_scaled

    def _prepare_data(self):
        self._get_data()
        self.X_train = self._scale_features_in_query_groups(self.X_train, self.query_ids_train)
        self.X_test = self._scale_features_in_query_groups(self.X_test, self.query_ids_test)
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)
        self.ys_train = torch.FloatTensor(self.y_train)
        self.ys_test = torch.FloatTensor(self.y_test)

    def _create_model(self):
        num_features = self.X_train.shape[1]
        self.model = ListNet(num_input_features=num_features, hidden_dim=self.listnet_hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = listnet_kl_loss

    def _calc_loss(self, ys_true, ys_pred):
        return listnet_kl_loss(ys_true, ys_pred)

    def _train_one_epoch(self):
        self._get_data()
        unique_queries = np.unique(self.query_ids_train)
        self._prepare_data()
        self._create_model()
        for query in unique_queries:
            query_indices = np.where(self.query_ids_train == query)[0]
            X_query = self.X_train[query_indices]
            y_query = self.y_train[query_indices]

            self.optimizer.zero_grad()
            y_pred = self.model(X_query).squeeze()
            loss = self._calc_loss(y_query, y_pred)
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self):
        with torch.no_grad():
            ndcg_scores = []
            unique_queries = np.unique(self.query_ids_test)
            for query in unique_queries:
                query_indices = np.where(self.query_ids_test == query)[0]
                X_query = self.X_test[query_indices]
                y_query = self.y_test[query_indices]

                y_pred = self.model(X_query).squeeze()
                ndcg_score = ndcg(y_query, y_pred, self.ndcg_top_k)
                ndcg_scores.append(ndcg_score.item())

            avg_ndcg = np.mean(ndcg_scores)
            return avg_ndcg

    def fit(self):
        ndcg_scores = []
        for epoch in range(self.n_epochs):
            self._train_one_epoch()
            ndcg_score = self._eval_test_set()
            ndcg_scores.append(ndcg_score)
            print(f"Epoch {epoch + 1}/{self.n_epochs}, NDCG@{self.ndcg_top_k}: {ndcg_score:.4f}")
        return ndcg_scores


if __name__ == '__main__':
    (train_df, test_df) = msrank_10k()
    sol = Solution(n_epochs=5, listnet_hidden_dim=20, lr=0.001, ndcg_top_k=10, root=train_df, test_root=test_df)
    sol.fit()
