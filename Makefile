# Detect OS-specific paths for venv Python and pip
ifeq ($(OS),Windows_NT)
	VENV_PY := .venv\Scripts\python.exe
	PIP := .venv\Scripts\pip.exe
else
	VENV_PY := .venv/bin/python
	PIP := .venv/bin/pip
endif

.PHONY: train venv

venv: .venv

.venv:
	python -m venv .venv
	$(VENV_PY) -m pip install -U pip
	$(PIP) install -r requirements.txt

# Build artifact when data or training script changes
train: .venv models/model.

models/model.pkl: data/spam.csv src/train.py
	$(VENV_PY) src/train.py

# Ensure models directory exists cross-platform
models:
	$(VENV_PY) -c "import os; os.makedirs('models', exist_ok=True)"

