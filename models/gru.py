import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super(GRU, self).__init__()
        # If num_layers = 1, PyTorch ignores dropout internally, so we conditionally set it.
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # We'll compute a simple attention from the GRU outputs:
        self.attn_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            out:         [batch_size, seq_len, hidden_size]
            attn_weights:[batch_size, seq_len, 1]
        """
        out, _ = self.gru(x)  # [B, S, H]
        # attention scores per time-step
        scores = self.attn_proj(out)            # [B, S, 1]
        attn_weights = torch.softmax(scores, dim=1)  # [B, S, 1]
        return out, attn_weights
