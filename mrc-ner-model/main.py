import os
from torch.utils.data import DataLoader
from dataset import *
from model import *
from torch.nn.modules import BCEWithLogitsLoss
import torch.optim as optim
from train import *
from transformers import AdamW
from model_config import model_config

def get_optimizer(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=2e-05,
                              eps=1e-08,)
    return optimizer

if __name__ == "__main__":
    # bert_path = "./ckpt_bert_base_uncased/"
    # vocab_file = os.path.join(bert_path, "vocab.txt")
    # json_path = "./data_dir/conll03/mrc-ner.train"

    # vocab_file = os.path.join(bert_path, "vocab.txt")
    config = model_config()
    tokenizer = BertWordPieceTokenizer(config.vocab_file)
    dataset = mrc_dataset(data_path=config.train_path, tokenizer=tokenizer, max_len=128)
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_to_max_length)

    val_dataset = mrc_dataset(data_path=config.dev_path, tokenizer=tokenizer, max_len=128)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=collate_to_max_length)

    model = BERT_MRC_DSC()
    bce_loss = BCEWithLogitsLoss(reduction="none")
    optimizer = get_optimizer(model)

    total_steps = ((len(dataset) / batch_size) + 1) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=2e-05, pct_start=0.0,
                final_div_factor=10000,
                total_steps=int(total_steps), anneal_strategy='linear'
            )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(model, dataloader, optimizer, scheduler, bce_loss, val_dataloader, num_epochs, device, config)
