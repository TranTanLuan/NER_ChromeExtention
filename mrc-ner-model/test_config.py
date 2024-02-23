import os

class test_config:
    def __init__(self):
        bert_path = "./ckpt_bert_base_uncased/"
        self.vocab_file = os.path.join(bert_path, "vocab.txt")
        self.pretrained_path = "./ckpt/2.pth"