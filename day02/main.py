import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 2, q_k_zip_dim: int = 128, v_zip_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.q_k_zip_dim = q_k_zip_dim
        self.v_zip_dim = v_zip_dim
        self.w_q = nn.Linear(self.d_model, self.q_k_zip_dim)
        self.w_k = nn.Linear(self.d_model, self.q_k_zip_dim)
        self.w_v_r = nn.Linear(self.d_model, self.v_zip_dim)
        self.w_v_l = nn.Linear(self.v_zip_dim, self.d_model)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # [max_seq_len, d_model]
        query = self.w_q(q) # [max_seq_len, q_k_zip_dim]
        key = self.w_k(k) # [max_seq_len, q_k_zip_dim]
        value = self.w_v_l(self.w_v_r(v))  # [max_seq_len, d_model]
        # 拆分多头
        query = query.view(query.shape[0], self.num_heads, -1).transpose(0, 1) # [num_heads, max_seq_len, q_k_zip_dim/num_heads]
        key = key.view(key.shape[0], self.num_heads, -1).transpose(0, 1) # [num_heads, max_seq_len, q_k_zip_dim/num_heads]
        value = value.view(value.shape[0], self.num_heads, -1).transpose(0, 1) # [num_heads, max_seq_len, d_model/num_heads]
        # 计算注意力权重
        attention_scores = query @ key.transpose(-2, -1) # [num_heads, max_seq_len, max_seq_len]
        attention_scores = attention_scores.softmax(dim=-1) # [num_heads, max_seq_len, max_seq_len]
        data = attention_scores @ value # [num_heads, max_seq_len, d_model/num_heads]
        data = data.transpose(0, 1).contiguous().view(q.shape[0], -1) # [max_seq_len, d_model]
        return data
        
class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.tensor, sublayer: nn.Module):
        return x + sublayer(x)
    
class FeedForwardNetWork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.hidden_layer_size = 500
        self.w_1 = nn.Linear(input_dim, self.hidden_layer_size)
        self.w_2 = nn.Linear(self.hidden_layer_size, output_dim)
    def forward(self, x: torch.Tensor):
        return self.w_2(self.w_1(x))
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = MultiHeadAttention(512, 2, 128, 128)
        self.residual_connection = nn.ModuleList([ResidualConnection() for i in range(2)])
        self.feed_forward_network = FeedForwardNetWork(512, 512)
    def forward(self, x: torch.Tensor):
        # 位置编码
        x = self.residual_connection[0](x, lambda x: self.attention_layer(x, x, x))
        x = self.residual_connection[1](x, self.feed_forward_network)
        return x