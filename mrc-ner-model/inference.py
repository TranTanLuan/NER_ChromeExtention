from dataset import *
from model import *
import os
from torch.utils.data import DataLoader
from typing import Tuple, List
from loss import *
from val import *
from torch.nn.modules import BCEWithLogitsLoss

class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})

def bmes_decode(char_label_list: List[Tuple[str, str]]) -> List[Tag]:
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0] # O, B, E, S, M

        # correct labels
        if idx + 1 == length and current_label == "B":
            current_label = "S"

        # merge chars
        if current_label == "O": #not entity
            idx += 1
            continue
        if current_label == "S": #same
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B": #begin
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1
            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else: #error with E
            break #lutr
            #raise Exception("Invalid Inputs")
    return tags

def extract_flat_spans(start_pred, end_pred, match_pred, label_mask, pseudo_tag = "TAG"):
    pseudo_input = "a"

    bmes_labels = ["O"] * len(start_pred) # default flag is O, is not entity
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}" # flag is B, token start
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{pseudo_tag}" # token end, if same token, flag is E

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end) #get min because using flat ner
        if match_pred[tmp_start][tmp_end]: # if is (start, end) pair
            if tmp_start != tmp_end: # start not similar end, set flag is M for element between B and E
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"M-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"S-{pseudo_tag}" #if start = end, flag is S

    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])

    return [(entity.begin, entity.end, entity.tag) for entity in tags]

def get_query_index_to_label_cate():
    return {1: "ORG", 2: "PER", 3: "LOC", 4: "MISC"}

def main():
    bert_path = "./ckpt_bert_base_uncased/"
    vocab_file = os.path.join(bert_path, "vocab.txt")
    json_path = "./data_dir/conll03/mrc-ner.test"

    vocab_file = os.path.join(bert_path, "vocab.txt")
    data_tokenizer = BertWordPieceTokenizer(vocab_file)
    dataset = mrc_dataset(data_path=json_path, tokenizer=data_tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    val_dataset = mrc_dataset(data_path="./data_dir/conll03/mrc-ner.dev", tokenizer=data_tokenizer, max_len=128)
    val_dataloader = DataLoader(val_dataset, batch_size=10,
                            collate_fn=collate_to_max_length)

    # load token
    with open(vocab_file, "r", encoding="utf8") as f:
        subtokens = [token.strip() for token in f.readlines()]
    idx2tokens = {}
    for token_idx, token in enumerate(subtokens):
        idx2tokens[token_idx] = token

    query2label_dict = get_query_index_to_label_cate()

    model = BERT_MRC_DSC()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("/content/drive/MyDrive/Jobs/Interview/2.pth"))
    bce_loss = BCEWithLogitsLoss(reduction="none")
    #val(model, val_dataloader, bce_loss, device)
    #exit()

    for batch in dataloader:
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, label_idx = batch
        #tokens: indexes in vocab (correspond to words)
        #token_type_ids: 0 for query, 1 for context
        #start_labels: 1 for start token pos, 0 for not
        #end_labels: 1 for end token pos, 0 for not
        #start_label_mask, end_label_mask: 0 for query and special token (cls, sep), 1 for context
        #match_labels: is matrix, each element at (start idx, end idx) is 1 for if start idx and end idx of entity, 0 for ...
        #label_idx: query order index (https://github.com/ShannonAI/mrc-for-flat-nested-ner/issues/39)
        #label_idx is index to get entity name, ex: {1: "ORG", 2: "PER", 3: "LOC", 4: "MISC"}, label_idx = 1

        attention_mask = (tokens != 0).long()
        #attention_mask: Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        #https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
        #https://huggingface.co/docs/transformers/glossary#attention-mask

        tokens = tokens.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)

        start_logits, end_logits, span_logits = model(tokens=tokens, token_types=token_type_ids,
                                                attention_mask=attention_mask)
        start_preds, end_preds, span_preds = start_logits > 0, end_logits > 0, span_logits > 0
        #get values > 0 because: labels is 0 or 1, using BCE_sigmoid loss, with sigmoid: negative -> 0, positive -> 1

        subtokens_idx_lst = tokens.cpu().numpy().tolist()[0]
        subtokens_lst = [idx2tokens[item] for item in subtokens_idx_lst] #convert idx in vocab to word
        label_cate = query2label_dict[label_idx.item()] #name of entity
        readable_input_str = data_tokenizer.decode(subtokens_idx_lst, skip_special_tokens=True) #decode tokens to ori text

        entities_info = extract_flat_spans(torch.squeeze(start_preds), torch.squeeze(end_preds),
                                            torch.squeeze(span_preds), torch.squeeze(attention_mask), pseudo_tag=label_cate)
        entity_lst = []

        if len(entities_info) != 0:
            for entity_info in entities_info:
                start, end = entity_info[0], entity_info[1]
                entity_string = " ".join(subtokens_lst[start: end])
                entity_string = entity_string.replace(" ##", "")
                entity_lst.append((start, end, entity_string, entity_info[2]))

        print("*="*10)
        print(f"Given input: {readable_input_str}")
        print(f"Model predict: {entity_lst}")
        # entity_lst is a list of (subtoken_start_pos, subtoken_end_pos, substring, entity_type)

if __name__ == "__main__":
    main()