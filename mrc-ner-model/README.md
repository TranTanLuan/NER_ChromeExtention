# Named Entity Recognition (NER)

## Installation
```
conda create -n env_name python=3.10.12
conda activate env_name
pip install transformers==4.35.2
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
- Can use colab (available environments) to train without installation.

## Training data
- Download in the paper "2019, BERT-MRC_A Unified MRC Framework for Named Entity Recognition"

## Training
- Adust config to train in mrc-ner-model/model_config.py:
    - bert_path: pretrained of BERT
    - batch_size
    - num_epochs
    - train_path: training data
    - dev_path: val data
- python main.py

## Testing
- Adust config to test in mrc-ner-model/test_config.py:
    - bert_path: pretrained of BERT
    - pretrained_path: pretrained of NER model to test
- python test.py "text to recognize entities"