.PHONY: all clean data train test lint

# Variables
PYTHON = python
DATA_DIR = data
CONFIG_FILE = configs/config.yaml

all: clean data train

clean:
	rm -rf $(DATA_DIR)/processed/*

setup:
	pip install -r requirements.txt
	python -m spacy download de_core_news_sm
	python -m spacy download en_core_web_sm

download_data:
	$(PYTHON) scripts/download_data.py --config $(CONFIG_FILE)

preprocess_data:
	$(PYTHON) scripts/preprocess_data.py --config $(CONFIG_FILE)

data: download_data preprocess_data

train:
	$(PYTHON) scripts/train.py --config $(CONFIG_FILE)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG_FILE)

translate:
	$(PYTHON) scripts/translate.py --config $(CONFIG_FILE)

test:
	pytest tests/

lint:
	black .
	isort .
	flake8 . 