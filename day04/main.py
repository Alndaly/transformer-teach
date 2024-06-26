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
        source_text = group[0].replace('english: ', '')
        source_token = source_tokenizer.encode(source_text)
        target_text = group[2].replace('mandarin: ', '')
        target_token = target_tokenizer.encode(target_text)
        
        encoder_input_token = torch.concat(
            [    
                torch.tensor(source_tokenizer.piece_to_id('<s>')).unsqueeze(0),
                torch.tensor(source_token),
                torch.tensor(source_tokenizer.piece_to_id('</s>')).unsqueeze(0),
                torch.full(size=(max_seq_len-2-len(source_token), ), fill_value=source_tokenizer.piece_to_id('<pad>'))
            ],
        )
        
        decoder_input_token = torch.concat(
            [    
                torch.tensor(target_tokenizer.piece_to_id('<s>')).unsqueeze(0),
                torch.tensor(target_token),
                torch.full(size=(max_seq_len-1-len(target_token), ), fill_value=target_tokenizer.piece_to_id('<pad>'))
            ],
        )
        
        decoder_label = torch.concat(
            [
                torch.tensor(target_token),
                torch.tensor(target_tokenizer.piece_to_id('</s>')).unsqueeze(0),
                torch.full(size=(max_seq_len-1-len(target_token), ), fill_value=target_tokenizer.piece_to_id('<pad>'))
            ]
        )
        
        return {
            'encoder_input_token': encoder_input_token,
            'decoder_input_token': decoder_input_token,
            'decoder_label': decoder_label,
        }
        
class Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer() for _ in range(6)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer() for _ in range(6)])
        self.source_embedding = InputEmbedding(source_tokenizer.get_piece_size(), 512)
        self.target_embedding = InputEmbedding(target_tokenizer.get_piece_size(), 512)
        self.position_encoding = PositionEncoding(512, 500)
        self.classifier = nn.Linear(512, target_tokenizer.get_piece_size())
    
    def encode(self, x: torch.Tensor):
        encoder_input_embedding = self.source_embedding(x)
        encoder_position_encoding = self.position_encoding(encoder_input_embedding)
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_position_encoding)
        return encoder_output
    
    def decode(self, encoder_output: torch.Tensor, x: torch.Tensor):
        decoder_input_embedding = self.target_embedding(x)
        decoder_position_encoding = self.position_encoding(decoder_input_embedding)
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_position_encoding, encoder_output)
        return decoder_output
        
    def project(self, decoder_output: torch.Tensor):
        output = self.classifier(decoder_output)
        output = torch.softmax(output, dim=-1)
        return output