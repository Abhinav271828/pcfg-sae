import torch
from sparsemax import Sparsemax
from torch import nn


class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, sparsemax):
        super(SAE, self).__init__()
        self.k = k                 # Optional[int]
        self.sparsemax = Sparsemax() if sparsemax else None

        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False), nn.ReLU())
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
    
    def forward(self, x):
        latent = self.encoder(x)
        if self.k:
            values, indices = torch.topk(latent, self.k)
            latent = torch.zeros_like(latent)
            latent.scatter_(-1, indices, values)
        if self.sparsemax:
            latent = self.sparsemax(latent)
        recon = self.decoder(latent)
        return latent, recon
