#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import torch
from torch.utils.data import DataLoader, Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_len
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx].item()
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(text, 
                                  return_tensors    = 'pt', 
                                  max_length        = self.max_length, 
                                  padding           = 'max_length', 
                                  truncation        = True)
        
        return {'text':           text,
                'input_ids':      encoding['input_ids'].flatten(), 
                'attention_mask': encoding['attention_mask'].flatten(), 
                'labels':          torch.tensor(label, dtype=torch.long)}
    

# ------------------------------------------------
# Dataset to concatenate text from different 
# columns and prepare it as input to a BERT model
# ------------------------------------------------
class BERTDataset(Dataset):
    def __init__(self, data, texts, labels, tokenizer, window_size, max_len):
        self.data               = data
        self.texts              = texts
        self.labels             = labels
        self.tokenizer          = tokenizer
        self.max_len            = max_len
        self.window_size        = window_size
        self.pretokenized_texts = self.pretokenize_texts()

    def pretokenize_texts(self):
        pretokenized_text   = []
        for index in self.texts.index:
            context_window           = self.data[['text']].loc[index-self.window_size:index].T
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
    


class GPTDataset(Dataset):
    def __init__(self, data, texts, labels, tokenizer, window_size, max_len):
        self.data               = data
        self.texts              = texts
        self.labels             = labels
        self.tokenizer          = tokenizer
        self.max_len            = max_len
        self.window_size        = window_size
        self.pretokenized_texts = self.pretokenize_texts()

    def pretokenize_texts(self):
        pretokenized_text   = []
        for index in self.texts.index:
            context_window           = self.data[['text']].loc[index-self.window_size:index].T
            add_especial_tokens      = context_window.apply(lambda row: " ".join(row.to_list()), axis=1).item()
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