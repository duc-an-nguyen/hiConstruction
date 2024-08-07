#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import torch
import torch.nn     as nn
from transformers   import BertModel, GPT2Model


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout=0.1):
        super(BERTClassifier, self).__init__()
        self.bert    = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x             = self.dropout(pooled_output)
            logits        = self.fc(x)
            return logits


class GPT2Classifier(nn.Module):
    def __init__(self, hidden_size, num_classes, max_seq_len, gpt_model_name, dropout=0.1):
        super(GPT2Classifier, self).__init__()
        self.gpt2model  = GPT2Model.from_pretrained(gpt_model_name)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size*max_seq_len, num_classes)

    def forward(self, input_ids, attention_mask):
        gpt_out, _      = self.gpt2model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        batch_size      = gpt_out.shape[0]
        linear_output   = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output