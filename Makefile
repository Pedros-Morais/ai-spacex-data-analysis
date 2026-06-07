# Makefile — atalhos de desenvolvimento para o projeto SpaceX Launch Success.
# Uso: `make install`, `make lint`, `make test`, `make train`, `make app`.

PYTHON ?= python
VENV   := .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help

.PHONY: help
help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: ## Cria o ambiente virtual (.venv) com Python 3.11
	$(PYTHON) -m venv $(VENV)

.PHONY: install
install: ## Instala dependências + o pacote em modo editável
	$(BIN)/python -m ensurepip --upgrade  # garante pip (venvs do uv vêm sem pip)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt
	$(BIN)/python -m pip install -e .

.PHONY: lint
lint: ## Roda ruff + black --check (sem warnings)
	$(BIN)/ruff check src tests scripts app
	$(BIN)/black --check src tests scripts app

.PHONY: format
format: ## Formata o código com black + ruff --fix
	$(BIN)/black src tests scripts app
	$(BIN)/ruff check --fix src tests scripts app

.PHONY: test
test: ## Roda os testes com cobertura (>= 80%)
	$(BIN)/pytest -q --cov=src/launch_success --cov-report=term-missing --cov-fail-under=80

.PHONY: ingest
ingest: ## Busca dados da API v4 da SpaceX -> data/processed/spacex_launches.csv
	$(BIN)/python scripts/run_ingestion.py

.PHONY: train
train: ## Roda o pipeline completo (treina, compara, escolhe, salva, SHAP)
	$(BIN)/python scripts/run_training.py

.PHONY: app
app: ## Sobe a aplicação Streamlit de inferência
	$(BIN)/streamlit run app/streamlit_app.py

.PHONY: clean
clean: ## Remove artefatos gerados (cache, cobertura, figuras, modelos)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	rm -f reports/figures/*.png models/*.joblib models/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
