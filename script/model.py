import os
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F

utility_path = os.path.split(os.path.realpath(__file__))[0] + "/utility_files/"

class SPROF_GO(nn.Module):
    def __init__(self, task, feature_dim=1024, hidden_dim=256, num_emb_layers=2, num_heads=8, dropout=0.1, device = torch.device('cpu')):
        super(SPROF_GO, self).__init__()

        # Child Matrix: CM_ij = 1 if the jth GO term is a subclass of the ith GO term
        self.CM = torch.tensor(ssp.load_npz(utility_path + task + "_CM.npz").toarray()).to(device)

        # Embedding layers
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(feature_dim, eps=1e-6)
                                        ,nn.Linear(feature_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )

        self.hidden_block = []
        for i in range(num_emb_layers - 1):
            self.hidden_block.extend([
                                      nn.LayerNorm(hidden_dim, eps=1e-6)
                                     ,nn.Dropout(dropout)
                                     ,nn.Linear(hidden_dim, hidden_dim)
                                     ,nn.LeakyReLU()
                                     ])
            if i == num_emb_layers - 2:
                self.hidden_block.extend([nn.LayerNorm(hidden_dim, eps=1e-6)])

        self.hidden_block = nn.Sequential(*self.hidden_block)

        # Self-attention pooling layer
        self.ATFC = nn.Sequential(
                                  nn.Linear(hidden_dim, 64)
                                 ,nn.LeakyReLU()
                                 ,nn.LayerNorm(64, eps=1e-6)
                                 ,nn.Linear(64, num_heads)
                                 )

        # Output layer
        self.label_size = {"MF":790, "BP":4766, "CC":667}[task] # terms with >= 50 samples in the training + validation sets
        self.output_block = nn.Sequential(
                                         nn.Linear(num_heads*hidden_dim, num_heads*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(num_heads*hidden_dim, eps=1e-6)
                                         ,nn.Dropout(dropout)
                                         ,nn.Linear(num_heads*hidden_dim, self.label_size)
                                         )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, h_V, mask, y = None):
        h_V = self.input_block(h_V)
        h_V = self.hidden_block(h_V)

        # Multi-head self-attention pooling
        att = self.ATFC(h_V)    # [B, L, num_heads]
        att = att.masked_fill(mask[:, :, None] == 0, -1e9)
        att = F.softmax(att, dim=1)
        att = att.transpose(1,2)   # [B, num_heads, L]
        h_V = att@h_V    # [B, num_heads, hidden_dim]
        h_V = torch.flatten(h_V, start_dim=1) # [B, num_heads*hidden_dim]

        h_V = self.output_block(h_V).sigmoid() # [B, label_size]

        # Hierarchical learning
        if self.training:
            a = (1 - y) * torch.max(h_V.unsqueeze(1) * self.CM.unsqueeze(0), dim = -1)[0] # the probability of a negative class should take the maximal probabilities of its subclasses
            b = y * torch.max(h_V.unsqueeze(1) * (self.CM.unsqueeze(0) * y.unsqueeze(1)), dim = -1)[0] # the probability of a positive class should take the maximal probabilities of its positive subclasses
            h_V = a + b
        else:
            h_V = torch.max(h_V.unsqueeze(1) * self.CM.unsqueeze(0), dim = -1)[0]  # [B, 1, label_size] * [1, label_size, label_size]

        return h_V.float() # [B, label_size]
