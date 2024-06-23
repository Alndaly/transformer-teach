import torch
from torch import nn
from rich import print
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from day01.main import get_or_build_tokenizer, PositionEncoding, InputEmbedding
from day03.main import TransformerDecoderLayer
from day02.main import TransformerEncoderLayer

source_tokenizer, target_tokenizer = get_or_build_tokenizer()

class MyData(Dataset):
    def __init__(self):
        super().__init__()
        ds_raw = load_dataset('swaption2009/20k-en-zh-translation-pinyin-hsk')
        self.ds_raw = ds_raw['train']
    def __len__(self):
        return int(len(self.ds_raw) / 5)
    def __getitem__(self, index: int):
        max_seq_len = 500
        group = self.ds_raw[index*5:index*5+4]['text']
        source = group[0].replace('english: ', '')
        source_token = source_tokenizer.encode(source)
        target = group[2].replace('mandarin: ', '')
        target_token = target_tokenizer.encode(target)
        
        source_token = torch.concat(
            [    
                torch.tensor(source_tokenizer.piece_to_id('<s>')).unsqueeze(0),
                torch.tensor(source_token),
                torch.tensor(source_tokenizer.piece_to_id('</s>')).unsqueeze(0),
                torch.full(size=(max_seq_len-2-len(source_token), ), fill_value=source_tokenizer.piece_to_id('<pad>'))
            ],
        )
        target_token = torch.concat(
            [    
                torch.tensor(target_tokenizer.piece_to_id('<s>')).unsqueeze(0),
                torch.tensor(target_token),
                torch.tensor(target_tokenizer.piece_to_id('</s>')).unsqueeze(0),
                torch.full(size=(max_seq_len-2-len(target_token), ), fill_value=target_tokenizer.piece_to_id('<pad>'))
            ],
        )
        
        return {
            'source': source_token,
            'target': target_token,
        }
        
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoderLayer()
        self.decoder = TransformerDecoderLayer()
        self.source_embedding = InputEmbedding(source_tokenizer.get_piece_size(), 512)
        self.target_embedding = InputEmbedding(target_tokenizer.get_piece_size(), 512)
        self.position_encoding = PositionEncoding(512, 500)
        self.classifier = nn.Linear(512, target_tokenizer.get_piece_size())
    def forward(self, x: torch.Tensor):
        encoder_input_embedding = self.source_embedding(x['source'])
        encoder_position_encoding = self.position_encoding(encoder_input_embedding)
        encoder_output = self.encoder(encoder_position_encoding)
        decoder_input_embedding = self.target_embedding(x['target'])
        decoder_position_encoding = self.position_encoding(decoder_input_embedding)
        decoder_output = self.decoder(decoder_position_encoding, encoder_output)
        output = self.classifier(decoder_output)
        output = torch.softmax(output, dim=-1)
        return output