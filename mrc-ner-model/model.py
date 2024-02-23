import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import torch

class BERT_MRC_DSC(nn.Module):
    def __init__(self, name="bert-base-uncased"):
        super(BERT_MRC_DSC, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = BertModel.from_pretrained(name)
        self.bert_output_size = self.model.pooler.dense.out_features
        self.start_index_pred = nn.Linear(self.bert_output_size, 1)
        self.end_index_pred = nn.Linear(self.bert_output_size, 1)
        self.start_end_matching = nn.Linear(self.bert_output_size*2, 1)
    def forward(self, tokens, token_types, attention_mask):
        # tokens: Indices of input sequence tokens in the vocabulary.
        # token_types: Segment token indices to indicate first and second portions of the inputs.
        # attention_mask: Mask to avoid performing attention on padding token indices

        bert_outs = self.model(input_ids=tokens, attention_mask=attention_mask, token_type_ids=token_types)
        context_matrix_E = bert_outs[0]
        sequence_length = context_matrix_E.shape[1]
        p_start = self.start_index_pred(context_matrix_E).squeeze(-1)
        p_end = self.end_index_pred(context_matrix_E).squeeze(-1)
        
        E_start = context_matrix_E.unsqueeze(2).expand(-1, -1, sequence_length, -1)
        E_end = context_matrix_E.unsqueeze(1).expand(-1, sequence_length, -1, -1)
        
        p_start_end = self.start_end_matching(torch.cat([E_start, E_end], 3)).squeeze(-1)
        return p_start, p_end, p_start_end