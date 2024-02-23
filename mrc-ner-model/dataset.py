import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from typing import List

def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    output.append(torch.stack([x[-1] for x in batch]))

    return output

class mrc_dataset(Dataset):
    def __init__(self, data_path, tokenizer: BertWordPieceTokenizer, max_len: int = 128):
        self.data = json.load(open(data_path, encoding="utf-8"))
        #self.data = self.data[:52]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_i = self.data[i]
        tokenizer = self.tokenizer

        qas_id = data_i.get("qas_id", "0.0")
        _, label_idx = qas_id.split(".")
        label_idx = torch.LongTensor([int(label_idx)])

        query = data_i["query"]
        context = data_i["context"]
        start_positions = data_i["start_position"] #token pos
        end_positions = data_i["end_position"] #token pos

        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions] #character pos
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions] #character pos

        query_context_encoding = tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_encoding.ids # tokens
        type_ids = query_context_encoding.type_ids # token types (0: query, 1: context)
        offsets = query_context_encoding.offsets #indicate (start character pos, end character pos) lists of tokens

        start_dict = {}
        end_dict = {}
        for idx in range(len(tokens)):
            if type_ids[idx] == 0: #query
                continue
            token_character_start, token_character_end = offsets[idx]
            if token_character_start == token_character_end == 0: #[CLS] token for the first seq or [SEP] token for separate
                continue
            start_dict[token_character_start] = idx
            end_dict[token_character_end] = idx

        new_start_positions = [start_dict[start] for start in start_positions] #token pos is changed when concat (query, context)
        new_end_positions = [end_dict[end] for end in end_positions] #token pos is changed when concat (query, context)

        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ] #mask: 0 for query and special token (cls, sep), 1 for context
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        for token_idx in range(len(tokens)):
            current_word_idx = query_context_encoding.words[token_idx]
            next_word_idx = query_context_encoding.words[token_idx+1] if token_idx+1 < len(tokens) else None
            prev_word_idx = query_context_encoding.words[token_idx-1] if token_idx-1 > 0 else None
            if prev_word_idx is not None and current_word_idx == prev_word_idx:
                start_label_mask[token_idx] = 0
            if next_word_idx is not None and current_word_idx == next_word_idx:
                end_label_mask[token_idx] = 0

        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))] #label: 1 for start token pos, 0 for not
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))] #label: 1 for end token pos, 0 for not

        tokens = tokens[: self.max_len]
        type_ids = type_ids[: self.max_len]
        start_labels = start_labels[: self.max_len]
        end_labels = end_labels[: self.max_len]
        start_label_mask = start_label_mask[: self.max_len]
        end_label_mask = end_label_mask[: self.max_len]

        sep_token = tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1
        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            label_idx
        ]