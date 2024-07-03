#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import torch
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------
# Dataset to concatenate text from different 
# columns and prepare it as input to a BERT model
# ------------------------------------------------
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text        = self.texts[idx]
        label       = self.labels[idx]
        encoding    = self.tokenizer.encode_plus(
            text                    = text,
            add_special_tokens      = False,
            max_length              = self.max_len,
            return_token_type_ids   = False,
            padding                 = 'max_length',
            return_attention_mask   = True,
            return_tensors          = 'pt',
        )
        
        return {
            'text':             text,
            'input_ids':        encoding['input_ids'].flatten(),
            'attention_mask':   encoding['attention_mask'].flatten(),
            'labels':           torch.tensor(label, dtype=torch.long)
        }
    
# ---------------------------------------------------
# Dataset that prepares the values ​​of a single column 
# and prepares them as input to a BERT model
# ---------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, window_size, max_len):
        self.texts              = texts
        self.labels             = labels
        self.tokenizer          = tokenizer
        self.max_len            = max_len
        self.window_size        = window_size
        self.pretokenized_texts = self.pretokenize_texts()

    def pretokenize_texts(self):
        window_size         = 10
        pretokenized_text   = []
        for index in self.texts.index:
            context_window           = self.texts.loc[index-window_size:index].T
            add_especial_tokens      = context_window.apply(lambda row: "[CLS] " + " [SEP] ".join(row.to_list()) + " [SEP]", axis=1).item()
            pretokenized_text.append(add_especial_tokens)
        return pretokenized_text

    def __len__(self):
        return len(self.pretokenized_texts)

    def __getitem__(self, idx):
        text        = self.pretokenized_texts[idx]
        label       = self.labels[idx]
        encoding    = self.tokenizer.encode_plus(
            text                    = text,
            truncation              = True,
            add_special_tokens      = False,
            max_length              = self.max_len,
            return_token_type_ids   = False,
            padding                 = 'max_length',
            return_attention_mask   = True,
            return_tensors          = 'pt',
        )
        
        return {
            'text':             text,
            'input_ids':        encoding['input_ids'].flatten(),
            'attention_mask':   encoding['attention_mask'].flatten(),
            'labels':           torch.tensor(label, dtype=torch.long)
        }
    
# ---------------------------------------------------
# Dataset that prepares data as input to a GPT2 model
# ---------------------------------------------------
class GPTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, window_size, max_len):
        self.texts              = texts
        self.labels             = labels
        self.tokenizer          = tokenizer
        self.max_len            = max_len
        self.window_size        = window_size
        self.pretokenized_texts = self.pretokenize_texts()

    def pretokenize_texts(self):
        window_size         = 10
        pretokenized_text   = []
        for index in self.texts.index:
            context_window           = self.texts.loc[index-window_size:index].T
            add_especial_tokens      = context_window.apply(lambda row: ", <|endoftext|>".join(row.to_list()) + " <|endoftext|>", axis=1).item()
            pretokenized_text.append(add_especial_tokens)
        return pretokenized_text

    def __len__(self):
        return len(self.pretokenized_texts)

    def __getitem__(self, idx):
        text        = self.pretokenized_texts[idx]
        label       = self.labels[idx]
        encoding    = self.tokenizer(
            text                    = text,
            truncation              = True,
            add_special_tokens      = False,
            max_length              = self.max_len,
            return_token_type_ids   = False,
            padding                 = 'max_length',
            return_attention_mask   = True,
            return_tensors          = 'pt',
        )
        
        return {
            'text':             text,
            'input_ids':        encoding['input_ids'].flatten(),
            'attention_mask':   encoding['attention_mask'].flatten(),
            'labels':           torch.tensor(label, dtype=torch.long)
        }