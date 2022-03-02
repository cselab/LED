#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import torch
import torch.nn as nn


# latent_state_dim
class beta_vae(nn.Module):
    def __init__(self, embedding_size=5):
        super(beta_vae, self).__init__()
        print("[beta_vae] Initializing Beta-VAE with embedding_size = {:}".
              format(embedding_size))
        self.fc_mu = nn.Linear(embedding_size, embedding_size)
        self.fc_std = nn.Linear(embedding_size, embedding_size)

        # raise ValueError("Only partly implemented and tested. Aborting...")

    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, latent_state):
        mu = self.fc_mu(latent_state)
        logvar = self.fc_std(latent_state)
        z = self.reparameterization_trick(mu, logvar)
        return z, mu, logvar
