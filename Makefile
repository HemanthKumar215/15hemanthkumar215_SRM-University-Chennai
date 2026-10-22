# Digital Twin for Type 2 Diabetes – Makefile
# Works on Linux/macOS with make; on Windows use `make` via Git Bash or run
# the commands directly in PowerShell.

PYTHON ?= python
PIP    ?= pip

.PHONY: setup data real-data pipeline hparam train train-baseline train-fusion app test clean help

help:  ## Show available targets
	@echo "Available targets:"
	@echo "  make setup          Install dependencies"
	@echo "  make data           Generate synthetic data (support --cohort_size)"
	@echo "  make real-data      Generate mock real datasets (AI-READI, CGMacros, Shanghai)"
	@echo "  make train          Train all models (baseline + fusion)"
	@echo "  make pipeline       Run 3-stage training (pretrain -> real fine-tune -> test)"
	@echo "  make hparam         Run Optuna hyperparameter optimization"
	@echo "  make app            Launch Streamlit dashboard"
	@echo "  make test           Run pytest suite"
	@echo "  make clean          Remove generated data and model files"

setup:  ## Install Python dependencies
	$(PIP) install -r requirements.txt

data:  ## Generate synthetic EHR + CGM data
	$(PYTHON) -m src.data_gen.generate_all

real-data:  ## Generate mock real-world datasets (AI-READI, CGMacros, Shanghai)
	$(PYTHON) -m src.data_real.mock_generator

pipeline:  ## Run 3-stage training pipeline (pretrain -> real fine-tune -> test)
	$(PYTHON) -m src.models.train_pipeline

hparam:  ## Run Optuna hyperparameter optimization
	$(PYTHON) -m src.models.hparam_search --n_trials 5

train-baseline:  ## Train LightGBM/XGBoost baseline
	$(PYTHON) -m src.models.train_baseline

train-fusion:  ## Train PyTorch fusion model
	$(PYTHON) -m src.models.train_fusion

train: train-baseline train-fusion  ## Train all models

app:  ## Launch Streamlit dashboard
	streamlit run app/streamlit_app.py

test:  ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

clean:  ## Remove generated artifacts
	@echo "Removing generated data and model files..."
	$(PYTHON) -c "import shutil, os; [shutil.rmtree(p, True) for p in ['data/ehr_profiles.csv','data/cgm_wearable.parquet','models','reports','logs'] if os.path.exists(p)]"
	@echo "Done."
