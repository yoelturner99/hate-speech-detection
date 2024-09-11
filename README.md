# DSTI DL Project A22 Cohort (Group 15)

The aim is to do sentiment analysis to identify situations of cyber-harassment, insults or abusive language in french. The model will be able to do binary classification of french text to *hateful* and *non-hateful*.

## Datasets used from Hugging Face (<https://huggingface.co/datasets/{dataset_name}>)

1) Paul/hatecheck-french
2) classla/FRENK-hate-en
3) odegiber/hate_speech18
4) Hate-speech-CNERG/hatexplain
5) tdavidson/hate_speech_offensive
6) limjiayi/hateful_memes_expanded
7) ucberkeley-dlab/measuring-hate-speech
8) tweets-hate-speech-detection/tweets_hate_speech_detection

Apart from the first dataset, all the of them are in english and therefore, we will use the API of Google Translate to translate all the datasets from english to french. Some of the datasets have multi-class labelling and they will be adjusted to have only two classes.

After preprocessing the data we come up with a total of **267841 of samples (50.9% *hateful* & 49.1% *non-hateful*)**

## Model - CamemBERT

CamemBERT is a state-of-the-art, large-scale language model specifically designed for French. It is based on Facebook's RoBERTa architecture, which is an optimized and enhanced version of the BERT (Bidirectional Encoder Representations from Transformers) model. We will leverages transfer learning through finetuning to do binary classification.

## Results on test data (20% of dataset)

|              | Non-hateful | Hateful    |
|--------------|-------------|------------|
| Non-hateful  | 12182       | 980        |
| Hateful      | 2686        | 10937      |

|   F1 Score   |   Recall    | Precision  | Accuracy |
|--------------|-------------|------------|----------|
|    86.27 %   |   86.31 %   |  86.94 %   | 86.31 %  |

## Venv Setup (python 3.10)

### With PIP

```bash
# Create venv
python -m venv .venv
# Activate venv in terminal
./.venv/Scripts/activate
# Install packages 
pip install -r ./requirements.txt
# To pull model and data from server
dvc pull
```

### With Conda

```bash
# Create conda env and install packages
conda env create --file environment.yml
# Activate conda env
conda activate mad-env
# To pull model and data from server
dvc pull
```

### CUDA for PyTorch

To do training and inference on GPU, download pytorch-cuda version (<https://pytorch.org/get-started/previous-versions/>) with the following commands in the terminal :

```bash
## To install pytorch-cuda version
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# OR
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
```

## Notebooks

### notebooks/01_dataset.ipynb

To download the raw datasets from HuggingFace.

### notebooks/02_preprocessing.ipynb

To preprocess and translate the data for finetuning and testing.

## Scripts

### scripts/finetuning.py

```bash
# To run the training script
python ./scripts/finetuning.py --data_path ./data/final/train_data.tsv --model_name camembert_mad_v0

# To run tensorboard for monitoring
tensorboard --logdir=runs
```

### scripts/testing.py

```bash
python ./scripts/testing.py --data_path ./data/final/test_data.tsv --model_dir ./models/camembert_mad_v1
```

## App

### app/main.py

FastAPI apps usually bind to 127.0.0.1 (localhost) or 0.0.0.0 (all network interfaces) by default. However, 0.0.0.0 is not meant to be used directly in a web browser. Therefore `http://localhost:8080` or `http://127.0.0.1:8080` should be used instead.

```bash
# To run the app locally
fastapi run app/main.py --port 8080
```

```bash
# Using CURL
curl -X POST "http://127.0.0.1:8080/predict/" -H "Content-Type: application/json" -d '{"text": "Your sample text here"}'

# Or in Powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8080/predict/" -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{"text": "Your sample text here"}'
```

## Hosting & Versioning

For the versioning and hosting of the code, data and models we combined:

- Github for code versioning
- DVC for data and model versioning
- Docker with a FastAPI app for deployment (yoelturner99/hate-speech-app:latest)

```bash
docker pull yoelturner99/hate-speech-app:latest

docker run -p 8080:8080 hate-speech-app
```

## Contributors

- Stuart Yoel TURNER
- Gaston AVELINE
- Alexandre SANOU
