import math
import torch
from torch import nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # num_embeddings: 词汇表大小，简单地说，如果你的模型需要处理10000个不同的单词或符号，那么num_embeddings就设置为10000。这样，每个单词或符号都会有一个对应的嵌入向量。
        # embedding_dim: 嵌入维度
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    def forward(self, x):
        x = self.embedding_layer(x)
        # 根据论文 还需要乘 sqrt(embedding_dim) 来变换embedding
        x = x * math.sqrt(self.embedding_dim)
        return x

class PositionEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_seq_len: int = 500) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=0.1)
        # 创建一个位置编码矩阵，大小为(max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # (max_seq_len, d_model / 2)
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).repeat(1, int(self.d_model / 2))
        # (max_seq_len, d_model / 2)
        div_term = (10000 ** (torch.arange(0, self.d_model, 2) / self.d_model)).unsqueeze(0).repeat(self.max_seq_len, 1) 
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        # 适配batch_size
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, d_model]
        # pe: [batch_size, max_seq_len, d_model] => [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        x = self.dropout(x)
        return x


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


class Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer() for _ in range(6)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer() for _ in range(6)])
        self.source_embedding = InputEmbedding(source_tokenizer.get_piece_size(), 512)
        self.target_embedding = InputEmbedding(target_tokenizer.get_piece_size(), 512)
        self.encoder_position_encode = PositionEncoding(512, 500)
        self.decoder_position_encode = PositionEncoding(512, 500)
        self.classifier = nn.Linear(512, target_tokenizer.get_piece_size())
    
    def encode(self, x: torch.Tensor):
        encoder_input_embedding = self.source_embedding(x)
        encoder_position_encoding = self.encoder_position_encode(encoder_input_embedding)
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_position_encoding)
        return encoder_output
    
    def decode(self, encoder_output: torch.Tensor, x: torch.Tensor):
        decoder_input_embedding = self.target_embedding(x)
        decoder_position_encoding = self.decoder_position_encode(decoder_input_embedding)
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_position_encoding, encoder_output)
        return decoder_output
        
    def project(self, decoder_output: torch.Tensor):
        output = self.classifier(decoder_output)
        output = torch.softmax(output, dim=-1)
        return output