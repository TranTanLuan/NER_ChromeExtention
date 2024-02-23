#conda create -n lutr_ner2 python=3.10.12
#pip install transformers==4.35.2
#pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
# from tokenizers import BertWordPieceTokenizer
# import os
from typing import Tuple, List
import torch
import numpy as np
from tqdm import tqdm

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

def get_data(tokens, type_ids, tokenizer, max_len=128):

    tokens = tokens[: max_len]
    type_ids = type_ids[: max_len]

    sep_token = tokenizer.token_to_id("[SEP]")
    if tokens[-1] != sep_token:
        tokens = tokens[: -1] + [sep_token]

    return [
        torch.LongTensor(tokens),
        torch.LongTensor(type_ids),
    ]

def split_tokens(context_encoding, len_query, max_len = 128):
    len_context = len(context_encoding)
    span = (max_len - len_query - 1)
    n = len_context // span
    list_context_encoding = []
    for i in range(n):
        list_context_encoding.append(context_encoding[i*span:(i+1)*span] + [102])
    if n * span < len_context:
        list_context_encoding.append(context_encoding[n*span:] + [102])
    return list_context_encoding

def get_character_pos(context_encoding_ori, start, end):
    out_start, out_end = 0, 0
    if start + 1 == end:
        out_start = context_encoding_ori.offsets[start][0]
        out_end = context_encoding_ori.offsets[start][1]
    else:
        out_start = context_encoding_ori.offsets[start][0]
        out_end = context_encoding_ori.offsets[end - 1][1]
    return out_start, out_end

def test_nermodel(context, vocab_file, model, data_tokenizer): #can process context with any length
    with open(vocab_file, "r", encoding="utf8") as f:
        subtokens = [token.strip() for token in f.readlines()]
    #subtokens: tokens in bert
    idx2tokens = {}
    for token_idx, token in enumerate(subtokens):
        idx2tokens[token_idx] = token

    query2label_dict = get_query_index_to_label_cate()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    list_query = [
        "organization entities are limited to named corporate, governmental, or other organizational entities.",
        "person entities are named persons or family.",
        "location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
        "examples of miscellaneous entities include events, nationalities, products and works of art."
    ]
    print("context: ", context)
    context_encoding_ori = data_tokenizer.encode(context) #encode context
    context_encoding = (context_encoding_ori.ids)[1:-1] #remove start token and end token to concat with query

    #encode list query
    list_query_encoding = []
    for i in range(len(list_query)):
        query_i = list_query[i]
        query_i_encoding = data_tokenizer.encode(query_i).ids
        list_query_encoding.append(query_i_encoding)

    #split context encoding into list of sub-context (len=128-32, 128: max to feed text to model, 32: max token of query)
    #len=128-32 to concat sub-context with query
    list_context_encoding = split_tokens(context_encoding, 32, 128) 

    len_list_context_encoding = []
    for i in range(len(list_context_encoding)):
        if i > 0:
            len_list_context_encoding.append(len_list_context_encoding[-1] + len(list_context_encoding[i]) - 1)
        else:
            len_list_context_encoding.append(len(list_context_encoding[i]) - 1)
    
    entity_lst = []
    # with each query, predict the context have which entity
    for i in range(len(list_query_encoding)):
        list_query_encoding_i = list_query_encoding[i]
        label_idx = i + 1

        for j in range(len(list_context_encoding)):
            list_context_encoding_j = list_context_encoding[j]
            tokens = list_query_encoding_i + list_context_encoding_j
            token_type_ids = [0]*len(list_query_encoding_i) + [1]*len(list_context_encoding_j)
            tokens, token_type_ids = get_data(tokens, token_type_ids, data_tokenizer)
            tokens = torch.unsqueeze(tokens, 0)
            token_type_ids = torch.unsqueeze(token_type_ids, 0)
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
            
            label_cate = query2label_dict[label_idx] #name of entity
            readable_input_str = data_tokenizer.decode(subtokens_idx_lst, skip_special_tokens=True) #decode tokens to ori text
            

            #postprocessing to extract start, end, entity name
            entities_info = extract_flat_spans(torch.squeeze(start_preds), torch.squeeze(end_preds),
                                                torch.squeeze(span_preds), torch.squeeze(attention_mask), pseudo_tag=label_cate)

            if len(entities_info) != 0:

                for entity_info in entities_info:
                    # start, end in query+subcontext
                    start, end = entity_info[0], entity_info[1]

                    entity_string = " ".join(subtokens_lst[start: end])
                    entity_string = entity_string.replace(" ##", "")
                    
                    start, end = start - len(list_query_encoding_i), end - len(list_query_encoding_i)
                    if j > 0:
                        #start, end are convertd from query+subcontext to ori context
                        start, end = start + len_list_context_encoding[j-1], end + len_list_context_encoding[j-1]
                    start, end = start + 1, end + 1
                    entity_lst.append((start, end, entity_string, entity_info[2])) #entity_info[2]: name of entity

    out_str = ""
    end_ls, content_ls = [], []
    for i in range(len(entity_lst)):
        #get start, end in character position
        start, end = get_character_pos(context_encoding_ori, entity_lst[i][0], entity_lst[i][1])
        out_str += f"{start},{end},{entity_lst[i][2]},{entity_lst[i][3]},{context[start:end]}\n"
        end_ls.append(end)
        content_ls.append(f"<mark>[{entity_lst[i][2]},{entity_lst[i][3]}]</mark>")
    end_idxes = np.argsort(end_ls) #sort list of end
    out_context = ""
    out_end = 0
    #insert mark to entity in context
    for i in tqdm(range(len(end_idxes))):
        start = 0
        end = end_ls[end_idxes[i]]
        content = content_ls[end_idxes[i]]
        if i > 0:
            start = len(context[:end_ls[end_idxes[i-1]]])
        out_context = out_context + context[start:end] + content
        out_end = end
    out_context = out_context + context[out_end:]
    # print("\nNER:\n", out_context)
    
    return out_context

def test_list_nermodel(list_context, vocab_file, model, data_tokenizer): #process list of context
    list_outs = []
    for i in range(len(list_context)):
        context = list_context[i]
        list_outs.append(test_nermodel(context, vocab_file, model, data_tokenizer))
    return list_outs

if __name__ == "__main__":
    # context = "China was acclaimed as the host country on 4 June 2019, as sole finishing bidder, days just prior to the 69th FIFA Congress in Paris, France.[6] The tournament was originally scheduled to be held from 16 June to 16 July 2023.[7] On 14 May 2022, the AFC announced that China would not host the tournament due to the COVID-19 pandemic and China's Zero-COVID policy.[8] Due to China's relinquishment of its hosting rights,[9][10] the AFC conducted a second round of bidding, with a deadline for submissions scheduled on 17 October 2022.[11] Four nations submitted bids: Australia, Indonesia, Qatar, and South Korea.[12] However, Australia subsequently withdrew in September 2022,[13] as did Indonesia on 15 October.[14] On 17 October, the AFC announced that Qatar had won the bid and would host the tournament.[3]"
    # context = "The ASEAN Football Federation Championship (less formally known as the AFF Championship or AFF Cup), currently known as the AFF Mitsubishi Electric Cup for sponsorship reasons, is the primary football tournament organized by the ASEAN Football Federation (AFF) for men's national teams in Southeast Asia. A biennial international competition, it is contested by the men's national teams of the AFF to determine the sub-continental champion of Southeast Asia. The competition has been held every two years since 1996, scheduled to be in the even-numbered year, except for 2007, and 2020 (which was postponed to 2021 due to the COVID-19 pandemic). It was felt that a close co-operation at the football level would improve the quality of sport across the region and make it more competitive at the Asian and world level. The AFF Championship title has been won by four national teams; Thailand have won seven titles, Singapore has four titles, Vietnam has two titles and Malaysia with one title. To date, Thailand and Singapore are the only teams in history to have won consecutive titles; Thailand in 2000 and 2002, 2014 and 2016 and also 2020 and 2022, and Singapore in 2004 and 2007. It is one of the most watched football tournaments in the region. The AFF Championship is also recognized as an 'A' international tournament by FIFA with FIFA ranking points being awarded since 1996.[1]"
    # context = "Vietnam is so beautiful, Thailand is so nice"

    #error at postprocessing with this sentence (bmes_decode, start index is negative)
    context = "During the mid-19th century, Leeds had constructed the impressive Grade I listed Leeds Town Hall, though the wealth which Manchester had acquired allowed them to retort by constructing striking architectural works of their own, such as the Grade I listed Manchester Town Hall. This served to establish the rivalry between the two cities even further."

    from tokenizers import BertWordPieceTokenizer
    import os
    import torch
    from .nermodel import BERT_MRC_DSC

    bert_path = "./myapp/ckpt_bert_base_uncased/"
    ckpt_path = "./myapp/ckpt/2.pth"

    vocab_file = os.path.join(bert_path, "vocab.txt")
    vocab_file = os.path.join(bert_path, "vocab.txt")
    data_tokenizer = BertWordPieceTokenizer(vocab_file)
    model = BERT_MRC_DSC()
    model.load_state_dict(torch.load(ckpt_path))
    out = test_nermodel(context, vocab_file, model, data_tokenizer)
    print(out)