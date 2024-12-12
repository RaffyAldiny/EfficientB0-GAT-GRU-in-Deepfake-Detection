# models/gru.py
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super(GRU, self).__init__()
        # If num_layers = 1, PyTorch's GRU ignores dropout internally, so we conditionally set it.
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            out: [batch_size, seq_len, hidden_size]
        """
        out, _ = self.gru(x)
        return out
