"""8673-parameter shared state encoder and candidate-conditioned margin head."""

import torch
from torch import nn


class MarginPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(241, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(23, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, state, condition):
        return self.head(torch.cat((self.encoder(state), condition), dim=-1)).squeeze(-1)

    @torch.jit.export
    def score_candidates(self, state, condition):
        embedding = self.encoder(state).reshape(1, 16).expand(condition.size(0), 16)
        return self.head(torch.cat((embedding, condition), dim=-1)).squeeze(-1)
