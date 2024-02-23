import os

class model_config:
    def __init__(self):
        bert_path = "./ckpt_bert_base_uncased/"
        self.vocab_file = os.path.join(bert_path, "vocab.txt")
        self.train_path = "./data_dir/conll03/mrc-ner.train"
        self.batch_size = 10
        self.num_epochs = 20
        self.dev_path= "./data_dir/conll03/mrc-ner.dev"
        self.out_ckpt = "./ckpt/"
        self.step_print = 100
        self.step_val = 1000