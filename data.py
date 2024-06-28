import torch
import sentencepiece as spm
from itertools import groupby
from torch.utils.data import Dataset
from datasets import load_dataset
from pathlib import Path

def split_list(lst, delimiter):
    # 使用 groupby 对列表进行分组，并过滤掉包含分隔符的分组
    return [list(group) for key, group in groupby(lst, lambda x: x == delimiter) if not key]

def get_all_sentences():
    ds_raw = load_dataset('swaption2009/20k-en-zh-translation-pinyin-hsk', split='train')['text']
    ds_raw = split_list(ds_raw, '--')
    for item in ds_raw:
        yield {
            'source': item[0].replace('english: ', ''),
            'target': item[2].replace('mandarin: ', ''),
            'pinyin': item[3].replace('pinyin: ', '')
        }

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
            character_coverage=1,
            model_prefix='target_tokenizer',
            pad_id=3
        )
    source_tokenizer = spm.SentencePieceProcessor(model_file='source_tokenizer.model')
    target_tokenizer = spm.SentencePieceProcessor(model_file='target_tokenizer.model')
    return source_tokenizer, target_tokenizer

source_tokenizer, target_tokenizer = get_or_build_tokenizer()

class TranslateData(Dataset):
    def __init__(self, max_seq_len = 500):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.ds_raw = load_dataset('swaption2009/20k-en-zh-translation-pinyin-hsk', split='train')
    def __len__(self):
        return int(len(self.ds_raw) / 5)
    def __getitem__(self, index: int):
        group = self.ds_raw[index*5:index*5+4]['text']
        source_text = group[0].replace('english: ', '')
        source_token = source_tokenizer.encode(source_text)
        target_text = group[2].replace('mandarin: ', '')
        target_token = target_tokenizer.encode(target_text)
        # print(source_text, source_token, source_tokenizer.id_to_piece(source_token))
        # print(target_text, target_token, target_tokenizer.id_to_piece(target_token))
        enc_num_padding_tokens = self.max_seq_len - len(source_token) - 2 # encoder要减掉<s>和</s>两个token
        dec_num_padding_tokens = self.max_seq_len - len(target_token) - 1 # docoder要减掉<s>一个token
        assert enc_num_padding_tokens >= 0 and dec_num_padding_tokens >= 0, f'句子太长了, {source_text}, {target_text}'
        # print(torch.tensor(source_token, dtype=torch.int64).shape)
        # print(torch.tensor(source_tokenizer.piece_to_id('<s>'), dtype=torch.int64).unsqueeze(0).shape)
        # print(torch.tensor(source_tokenizer.piece_to_id('</s>'), dtype=torch.int64).unsqueeze(0).shape)
        # print(torch.full(size=(enc_num_padding_tokens, ), dtype=torch.int64, fill_value=source_tokenizer.piece_to_id('<pad>')).shape)
        # 增加<s>和</s>
        encoder_input_token = torch.concat(
            (   
                torch.tensor(source_tokenizer.piece_to_id('<s>'), dtype=torch.int64).unsqueeze(0),
                torch.tensor(source_token, dtype=torch.int64),
                torch.tensor(source_tokenizer.piece_to_id('</s>'), dtype=torch.int64).unsqueeze(0),
                torch.full(size=(enc_num_padding_tokens, ), fill_value=source_tokenizer.piece_to_id('<pad>'), dtype=torch.int64)
            ),
            dim=0
        )
        # 增加<s>
        decoder_input_token = torch.concat(
            [    
                torch.tensor(target_tokenizer.piece_to_id('<s>'), dtype=torch.int64).unsqueeze(0),
                torch.tensor(target_token, dtype=torch.int64),
                torch.full(size=(dec_num_padding_tokens, ), fill_value=target_tokenizer.piece_to_id('<pad>'), dtype=torch.int64)
            ],
        )
        # 增加</s>
        decoder_label = torch.concat(
            [
                torch.tensor(target_token, dtype=torch.int64),
                torch.tensor(target_tokenizer.piece_to_id('</s>'), dtype=torch.int64).unsqueeze(0),
                torch.full(size=(dec_num_padding_tokens, ), fill_value=target_tokenizer.piece_to_id('<pad>'), dtype=torch.int64)
            ]
        )
        
        # False 表示遮蔽，True 表示不遮蔽
        decoder_mask = (decoder_input_token == target_tokenizer.piece_to_id('<pad>')) | (casual_mask(decoder_input_token.shape[0]) == 0) == False
        
        return {
            'encoder_input_token': encoder_input_token, # [max_seq_len]
            'decoder_input_token': decoder_input_token, # [max_seq_len]
            'decoder_label': decoder_label, # [max_seq_len]
            'decoder_mask': decoder_mask,
            'source_text': source_text,
            'target_text': target_text
        }

def casual_mask(size):
    mask = torch.tril(torch.ones(size, size, dtype=torch.int))
    return mask