import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, q_k_zip_dim: int = 128, v_zip_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # Make sure d_model is divisible by h
        assert d_model % num_heads == 0, "d_model is not divisible by h"
        self.q_k_zip_dim = q_k_zip_dim
        self.v_zip_dim = v_zip_dim
        # self.w_q = nn.Linear(self.d_model, self.q_k_zip_dim)
        # self.w_k = nn.Linear(self.d_model, self.q_k_zip_dim)
        # self.w_v_r = nn.Linear(self.d_model, self.v_zip_dim)
        # self.w_v_l = nn.Linear(self.v_zip_dim, self.d_model)
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(d_model, d_model)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # [batch_size, max_seq_len, d_model]
        query = self.w_q(q) # [batch_size, max_seq_len, q_k_zip_dim]
        key = self.w_k(k) # [batch_size, max_seq_len, q_k_zip_dim]
        # value = self.w_v_l(self.w_v_r(v))  # [batch_size, max_seq_len, d_model]
        value = self.w_v(v)  # [batch_size, max_seq_len, d_model]
        # 拆分多头
        query = query.view(query.shape[0], query.shape[1], self.num_heads, -1).transpose(1, 2) # [batch_size, num_heads, max_seq_len, q_k_zip_dim/num_heads]
        key = key.view(key.shape[0], key.shape[1], self.num_heads, -1).transpose(1, 2) # [batch_size, num_heads, max_seq_len, q_k_zip_dim/num_heads]
        value = value.view(value.shape[0], value.shape[1], self.num_heads, -1).transpose(1, 2) # [batch_size, num_heads, max_seq_len, d_model/num_heads]
        # 计算注意力权重
        attention_scores = query @ key.transpose(-2, -1) # [batch_size, num_heads, max_seq_len, max_seq_len]
        attention_scores = attention_scores.softmax(dim=-1) # [batch_size, num_heads, max_seq_len, max_seq_len]
        data = attention_scores @ value # [batch_size, num_heads, max_seq_len, d_model/num_heads]
        data = data.transpose(0, 1).contiguous().view(q.shape[0], q.shape[1], -1) # [batch_size, max_seq_len, d_model]
        data = self.w_o(data)
        return data
        
class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x: torch.tensor, sublayer: nn.Module):
        return x + self.dropout(sublayer(x))
    
class FeedForwardNetWork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.hidden_layer_size = 2048
        self.dropout = nn.Dropout(p=0.1)
        self.w_1 = nn.Linear(input_dim, self.hidden_layer_size)
        self.w_2 = nn.Linear(self.hidden_layer_size, output_dim)
    def forward(self, x: torch.Tensor):
        x = torch.relu(self.w_1(x))
        x = torch.relu(self.w_2(x))
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = MultiHeadAttention(512, 8, 128, 128)
        self.residual_connection = nn.ModuleList([ResidualConnection() for i in range(2)])
        self.feed_forward_network = FeedForwardNetWork(512, 512)
        self.layer_norm = nn.LayerNorm(512, eps=10**-6)
    def forward(self, x: torch.Tensor):
        # 位置编码
        x = self.residual_connection[0](x, lambda x: self.attention_layer(x, x, x))
        x = self.layer_norm(x)
        x = self.residual_connection[1](x, self.feed_forward_network)
        x = self.layer_norm(x)
        return x