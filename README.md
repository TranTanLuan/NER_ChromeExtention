# Named Entity Recognition (NER) - Chrome Extension

## Introduction
The project builds a Chrome extension, it reads the content of the page (a user visits) and sends that content to a server (Django backend). That server extracts entities and sends the results back to the extension. The extension displays the results on the webpage.

## Project structure
- backend: src for Django backend
- extension: src for Chrome extension
- mrc-ner-model: src for training a NER model
- you can read in each folder to understand clearly

## Installation
```
conda create -n env_name python=3.10.12
conda activate env_name
pip install Django
pip install transformers==4.35.2
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Prepare
- "cd mrc-ner-model/" to train a ner model, save the best checkpoint
- move that checkpoint to "backend/myapp/ckpt/" folder
- put pretrained BERT at "backend/myapp/ckpt_bert_base_uncased/"

## Run
- Go to "backend" folder: cd backend
- Run the server: python manage.py runserver
- Go to extension link: chrome://extensions/, load unpacked "extension" folder
- Go to the webpage which you want to extract entities: use extension (watch demo to be clear)

## Demo video
- Demo video: https://www.youtube.com/watch?v=QN19uU4AWuw