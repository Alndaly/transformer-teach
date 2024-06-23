import torch
from torch import nn
from day02.main import ResidualConnection, FeedForwardNetWork, MultiHeadAttention

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 2, q_k_zip_dim: int = 128, v_zip_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_k_zip_dim = q_k_zip_dim
        self.v_zip_dim = v_zip_dim
        self.w_q = nn.Linear(d_model, self.q_k_zip_dim)
        self.w_k = nn.Linear(d_model, self.q_k_zip_dim)
        self.w_v_r = nn.Linear(d_model, self.v_zip_dim)
        self.w_v_l = nn.Linear(self.v_zip_dim, d_model)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # [max_seq_len, d_model]
        query = self.w_q(q) # [max_seq_len, q_k_zip_dim]
        key = self.w_k(k) # [max_seq_len, q_k_zip_dim]
        value = self.w_v_l(self.w_v_r(v)) # [max_seq_len, d_model]
        # 多头分割
        query = query.view(q.shape[0], self.num_heads, -1).transpose(0, 1) # [num_heads, max_seq_len, q_k_zip_dim/num_heads]
        key = key.view(k.shape[0], self.num_heads, -1).transpose(0, 1) # [num_heads, max_seq_len, q_k_zip_dim/num_heads]
        value = value.view(v.shape[0], self.num_heads, -1).transpose(0, 1) # [num_heads, max_seq_len, d_model/num_heads]
        attention_scores = query @ key.transpose(-2, -1) # [num_heads, max_seq_len, max_seq_len]
        mask = torch.triu(torch.ones(query.shape[0], query.shape[0]), diagonal=1).unsqueeze(0) # [1, max_seq_len, max_seq_len]
        attention_scores.masked_fill_(mask == 1, float('-inf')) # [num_heads, max_seq_len, max_seq_len]
        attention_scores = attention_scores.softmax(dim=-1) # [num_heads, max_seq_len, max_seq_len]
        data = attention_scores @ value # [num_heads, max_seq_len, d_model/num_heads]
        data = data.transpose(0, 1).contiguous().view(q.shape[0], -1) # [max_seq_len, d_model]
        return data

class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_attention_layer = MaskedMultiHeadAttention(512, 2, 128, 128)
        self.attention_layer = MultiHeadAttention(512, 2, 128, 128)
        self.residual_connection = nn.ModuleList([ResidualConnection() for _ in range(3)])
        self.feed_forward_network = FeedForwardNetWork(512, 512)
    def forward(self, x, encoder_output):
        x = self.residual_connection[0](x, lambda x: self.masked_attention_layer(x, x, x))
        x = self.residual_connection[1](x, lambda x: self.attention_layer(x, encoder_output, encoder_output))
        x = self.residual_connection[2](x, self.feed_forward_network)
        return x