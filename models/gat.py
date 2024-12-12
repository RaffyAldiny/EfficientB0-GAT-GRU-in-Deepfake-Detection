# models/gat.py
import torch
from torch_geometric.nn import GATConv
import torch.nn as nn

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=True, dropout=0.6)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x, edge_index):
        """
        Args:
            x: [N, in_channels]
            edge_index: [2, E]
        Returns:
            out: [N, heads * out_channels]
        """
        x = self.gat(x, edge_index)  # [N, heads * out_channels]
        x = self.elu(x)
        x = self.dropout(x)
        return x
