from django.http import JsonResponse
from .test_nermodel import test_list_nermodel
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

# Create your views here.
def process_text_from_extension(request):
    text = request.GET.get('text', None)

    text = text.replace("\n", "")

    splited_text = text.split("[===]")
    splited_text = splited_text[1:]

    data = {
        'summary': test_list_nermodel(splited_text, vocab_file, model, data_tokenizer),
        'raw': 'Successful',
    }

    return JsonResponse(data)