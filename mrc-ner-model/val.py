from loss import *
import numpy as np
from tqdm import tqdm

def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels):
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()
    return torch.stack([tp, fp, fn])

def val(model, dataloader, bce_loss, device):
    model.eval()
    list_tp, list_fp, list_fn = [], [], []
    for batch in (dataloader):
        tokens, token_type_ids, g_start, g_end, g_start_mask, g_end_mask, g_start_end, _ = batch
        attention_mask = (tokens != 0).long()

        tokens = tokens.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        g_start = g_start.to(device)
        g_end = g_end.to(device)
        g_start_mask = g_start_mask.to(device)
        g_end_mask = g_end_mask.to(device)
        g_start_end = g_start_end.to(device)

        p_start, p_end, p_start_end = model(tokens=tokens, token_types=token_type_ids,
                                            attention_mask=attention_mask)
        start_loss, end_loss, span_loss = mrc_loss(bce_loss=bce_loss, p_start=p_start,
                 p_end=p_end,
                 p_start_end=p_start_end,
                 g_start=g_start,
                 g_end=g_end,
                 g_start_mask=g_start_mask,
                 g_end_mask=g_end_mask,
                 g_start_end=g_start_end)
        total_loss = 1.0/3 * start_loss + 1.0/3 * end_loss + 1.0/3 * span_loss
        list_metrics = query_span_f1(p_start, p_end, p_start_end, g_start_mask, g_end_mask, g_start_end)
        list_tp.append(list_metrics[0])
        list_fp.append(list_metrics[1])
        list_fn.append(list_metrics[2])

    span_tp = torch.sum(torch.Tensor(list_tp))
    span_fp = torch.sum(torch.Tensor(list_fp))
    span_fn = torch.sum(torch.Tensor(list_fn))

    span_recall = span_tp / (span_tp + span_fn + 1e-10)
    span_precision = span_tp / (span_tp + span_fp + 1e-10)
    span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
    print("f1, recall, precision: ", span_f1, span_recall, span_precision)
    model.train()