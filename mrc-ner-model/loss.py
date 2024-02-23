import torch

def mrc_loss(bce_loss, p_start, p_end, p_start_end,
                    g_start, g_end, g_start_end, g_start_mask, g_end_mask):
    #p_start, p_end, p_start_end: Start Index Prediction, End Index Prediction, Start-End Matching Prediction
    #g_start, g_end, g_start_end: Start Index GT, End Index GT, Start-End Matching GT
    #g_start_mask, g_end_mask: 0 for query and special token (cls, sep), 1 for context

    b, seq_len = p_start.shape

    g_start_mask_1dim = g_start_mask.view(-1).float()
    g_end_mask_1dim = g_end_mask.view(-1).float()
    # each (start idx, end idx) can be context or query
    g_start_mask_row = g_start_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    g_end_mask_col = g_end_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    g_start_end_mask = g_start_mask_row & g_end_mask_col
    g_start_end_mask = torch.triu(g_start_end_mask, 0)  # start idx < end idx

    float_g_start_end_mask = g_start_end_mask.view(b, -1).float()

    start_loss = bce_loss(p_start.view(-1), g_start.view(-1).float())
    start_loss = (start_loss * g_start_mask_1dim).sum() / g_start_mask_1dim.sum()
    end_loss = bce_loss(p_end.view(-1), g_end.view(-1).float())
    end_loss = (end_loss * g_end_mask_1dim).sum() / g_end_mask_1dim.sum()
    span_loss = bce_loss(p_start_end.view(b, -1), g_start_end.view(b, -1).float())
    span_loss = span_loss * float_g_start_end_mask
    span_loss = span_loss.sum() / (float_g_start_end_mask.sum() + 1e-10)

    return start_loss, end_loss, span_loss