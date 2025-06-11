# from __future__ import annotations
#
# from typing import Any
#
# import mlflow
# import numpy as np
# import pandas as pd
# import torch
# from torch import nn
# import pytorch_lightning as pl
#
# from train.registries.mlflow_registry import register_model
# from train.trainers.base import TabularTrainerMixin
#
#
# class _LSTM(pl.LightningModule):
#     def __init__(self, n_features: int):
#         super().__init__()
#         self.lstm = nn.LSTM(n_features, 64, 2, batch_first=True)
#         self.fc = nn.Linear(64, 1)
#         self.loss = nn.L1Loss()
#
#     def forward(self, x):
#         return self.fc(self.lstm(x)[0][:, -1, :])
#
#     def training_step(self, batch, _):
#         x, y = batch
#         l = self.loss(self(x).squeeze(), y)
#         self.log("loss", l)
#         return l
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-3)
#
#
# class LightningLSTMTrainer(TabularTrainerMixin):
#     def __init__(self, cfg: dict[str, Any]):
#         self.cfg = cfg
#         self._net: _LSTM | None = None
#
#     def preprocess(self, df: pd.DataFrame):
#         return df, None  # windowing inside fit()
#
#     # Building windowed dataset
#     def _dataloader(self, df: pd.DataFrame, k: int):
#         X, y = [], []
#         for i in range(len(df) - k):
#             win = df.iloc[i : i + k]
#             X.append(win.drop("close", axis=1).values)
#             y.append(win["close"].iloc[-1])
#         X = torch.tensor(np.array(X), dtype=torch.float32)
#         y = torch.tensor(np.array(y), dtype=torch.float32)
#         return torch.utils.data.DataLoader(
#             torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True
#         )
#
#     def fit(self, df, _):
#         k = self.cfg.get("seq_len", 30)
#         dl = self._dataloader(df, k)
#         self._net = _LSTM(df.shape[1] - 1)
#         trainer = pl.Trainer(max_epochs=20, logger=False, enable_progress_bar=False)
#         trainer.fit(self._net, dl)
#         register_model(self._net, self.cfg)
#         return self._net
#
#     def predict(self, X):
#         if self._net is None:
#             raise RuntimeError("Model not fitted")
#         return np.zeros(len(X))  # demo‑stub

