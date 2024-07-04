import torch
import math
import sentencepiece as spm
from torch import nn
from rich import print
from pathlib import Path
from datasets import load_dataset
from itertools import groupby

def get_all_sentences():
    ds_raw = load_dataset('swaption2009/20k-en-zh-translation-pinyin-hsk', split='train')['text']
    data = [list(group) for key, group in groupby(ds_raw, lambda x: x == '--') if not key]
    for item in data:
        yield {
            'source': item[0].replace('english: ', ''),
            'target': item[2].replace('mandarin: ', ''),
            'pinyin': item[3].replace('pinyin: ', '')
        }
    

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
    
def get_or_build_tokenizer():
    if not Path.exists(Path('source_tokenizer.model')):
        spm.SentencePieceTrainer.train(
            sentence_iterator=map(lambda x: x['source'], get_all_sentences()),
            model_type='BPE',
            vocab_size=20000,
            character_coverage=1,
            model_prefix='source_tokenizer',
            pad_id=3
        )
    if not Path.exists(Path('target_tokenizer.model')):
        spm.SentencePieceTrainer.train(
            sentence_iterator=map(lambda x: x['target'], get_all_sentences()),
            model_type='BPE',
            vocab_size=10000,
            character_coverage=0.9995,
            model_prefix='target_tokenizer',
            pad_id=3
        )
    source_tokenizer = spm.SentencePieceProcessor(model_file='source_tokenizer.model')
    target_tokenizer = spm.SentencePieceProcessor(model_file='target_tokenizer.model')
    return source_tokenizer, target_tokenizer

if __name__ == '__main__':
    source_tokenizer, _ = get_or_build_tokenizer()
    ids = source_tokenizer.encode('Who are you?')
    ids = torch.tensor(ids)
    max_seq_len = 500
    ids = torch.concat(
        [    
            torch.tensor(source_tokenizer.piece_to_id('<s>')).unsqueeze(0),
            ids,
            torch.tensor(source_tokenizer.piece_to_id('</s>')).unsqueeze(0),
            torch.full(size=(max_seq_len-2-len(ids), ), fill_value=source_tokenizer.piece_to_id('<pad>'))
        ],
    )
    embdding = InputEmbedding(source_tokenizer.get_piece_size())
    output = embdding(ids)
    position_encoding = PositionEncoding()
    output = position_encoding(output)