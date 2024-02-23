from loss import *
from val import *
from tqdm import tqdm
import os

def train(model, dataloader, optimizer, scheduler, bce_loss, val_dataloader, num_epochs, device, config):
    out_path = config.out_ckpt
    os.makedirs(out_path, exist_ok=True)
    for i_epoch in range(num_epochs):
        model.train()
        count = 0
        for batch in tqdm(dataloader):
            tokens, token_type_ids, g_start, g_end, g_start_mask, g_end_mask, g_start_end, _ = batch
            attention_mask = (tokens != 0).long()

            # zero the parameter gradients
            optimizer.zero_grad()

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

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            count += 1
            if count % config.step_print == 0:
                print("\ntotal_loss: ", total_loss)
                print("lr: ", optimizer.param_groups[0]["lr"])
            if count % config.step_val == 0:
                val(model, val_dataloader, bce_loss, device)
        print("===============epoch: ", i_epoch)
        val(model, val_dataloader, bce_loss, device)
        torch.save(model.state_dict(), out_path + f"{i_epoch}.pth")
        print("===============")