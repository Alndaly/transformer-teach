import torch
from torch import nn
from day02.main import ResidualConnection, FeedForwardNetWork, MultiHeadAttention

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, q_k_zip_dim: int = 128, v_zip_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_k_zip_dim = q_k_zip_dim
        self.v_zip_dim = v_zip_dim
        # self.w_q = nn.Linear(d_model, q_k_zip_dim)
        # self.w_k = nn.Linear(d_model, q_k_zip_dim)
        # self.w_v_r = nn.Linear(d_model, v_zip_dim)
        # self.w_v_l = nn.Linear(v_zip_dim, d_model)
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(d_model, d_model)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # [batch_size, max_seq_len, d_model]
        query = self.w_q(q) # [batch_size, max_seq_len, q_k_zip_dim]
        key = self.w_k(k) # [batch_size, max_seq_len, q_k_zip_dim]
        # value = self.w_v_l(self.w_v_r(v)) # [batch_size, max_seq_len, d_model]
        value = self.w_v(v)
        # 多头分割
        query = query.view(query.shape[0], query.shape[1], self.num_heads, -1).transpose(1, 2) # [batch_size, num_heads, max_seq_len, q_k_zip_dim/num_heads]
        key = key.view(key.shape[0], key.shape[1], self.num_heads, -1).transpose(1, 2) # [batch_size, num_heads, max_seq_len, q_k_zip_dim/num_heads]
        value = value.view(value.shape[0], value.shape[1], self.num_heads, -1).transpose(1, 2) # [batch_size, num_heads, max_seq_len, d_model/num_heads]
        attention_scores = query @ key.transpose(-2, -1) # [batch_size, num_heads, max_seq_len, max_seq_len]
        mask = torch.triu(torch.ones(q.shape[1], q.shape[1]), diagonal=1).unsqueeze(0).unsqueeze(0) # [1, 1, max_seq_len, max_seq_len]
        attention_scores.masked_fill_(mask == 1, float('-inf')) # [batch_size, num_heads, max_seq_len, max_seq_len]
        attention_scores = attention_scores.softmax(dim=-1) # [batch_size, num_heads, max_seq_len, max_seq_len]
        data = attention_scores @ value # [batch_size, num_heads, max_seq_len, d_model/num_heads]
        data = data.transpose(0, 1).contiguous().view(q.shape[0], q.shape[1], -1) # [batch_size, max_seq_len, d_model]
        data = self.w_o(data)
        return data

class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_attention_layer = MaskedMultiHeadAttention(512, 8, 128, 128)
        self.attention_layer = MultiHeadAttention(512, 8, 128, 128)
        self.residual_connection = nn.ModuleList([ResidualConnection() for _ in range(3)])
        self.feed_forward_network = FeedForwardNetWork(512, 512)
        self.layer_norm = nn.LayerNorm(512)
    def forward(self, x, encoder_output):
        x = self.residual_connection[0](x, lambda x: self.masked_attention_layer(x, x, x))
        x = self.layer_norm(x)
        x = self.residual_connection[1](x, lambda x: self.attention_layer(x, encoder_output, encoder_output))
        x = self.layer_norm(x)
        x = self.residual_connection[2](x, self.feed_forward_network)
        x = self.layer_norm(x)
        return x