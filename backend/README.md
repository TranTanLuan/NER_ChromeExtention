# Django backend
## Introduction
The Django backend use a NER model to extract entities.

## Installation
```
conda create -n env_name python=3.10.12
conda activate env_name
pip install Django
pip install transformers==4.35.2
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Run
- Run the server:
```
python manage.py runserver
```
- Test the NER model:
```
python -m myapp.test_nermodel
```
- Add an url to call a processing fuction, go to "backend/backend/urls.py", for example:
    - At line 22, "path("process_text_from_extension/", views.process_text_from_extension)"
    - when connect to "process_text_from_extension/" endpoint, it will call "views.process_text_from_extension" function
- "backend/myapp/views.py": modify ckpt path, the processing function

## How to create Django backend
```
django-admin startproject backend
python manage.py startapp myapp
```