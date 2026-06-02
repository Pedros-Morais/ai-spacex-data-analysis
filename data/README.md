# Dados — proveniência e schema

Este diretório guarda os dados do projeto. Há **dois caminhos** de obtenção,
com a mesma estrutura de colunas (schema), descritos abaixo.

## Estrutura

```
data/
├── raw/                         # JSON cru consolidado da API (gerado; .gitignored)
│   └── spacex_launches_raw.json
├── processed/
│   └── spacex_launches.csv      # dataset tabular usado pelo pipeline (versionado)
└── README.md                    # este arquivo
```

## Origem dos dados

### 1) Caminho canônico — API pública v4 da SpaceX (dado **real**)

Módulo: [`src/launch_success/data/ingestion.py`](../src/launch_success/data/ingestion.py)
(cliente em [`spacex_client.py`](../src/launch_success/data/spacex_client.py)).

Endpoints consumidos de `https://api.spacexdata.com/v4`:

| Endpoint        | Uso                                                        |
|-----------------|------------------------------------------------------------|
| `GET /launches` | lista de lançamentos (cada um traz o array `cores`)        |
| `GET /rockets`  | resolve `rocket id` → nome (Falcon 1 / 9 / Heavy)          |
| `GET /payloads` | resolve `payload id` → `mass_kg`, `orbit`                  |
| `GET /launchpads` | resolve `launchpad id` → nome do local                   |

A ingestão resolve os `id`s, produz **uma linha por lançamento**, salva o JSON
cru em `data/raw/` e o CSV processado em `data/processed/spacex_launches.csv`.

Rode com:

```bash
make ingest          # ou: python scripts/run_ingestion.py
```

### 2) Fallback reprodutível — dataset **sintético** calibrado

Módulo: [`src/launch_success/data/synthetic.py`](../src/launch_success/data/synthetic.py).

Como a API pode estar **indisponível** no ambiente de correção, o repositório
inclui um CSV versionado (`data/processed/spacex_launches.csv`) gerado pelo
gerador sintético, para o pipeline rodar *out-of-the-box*. O enunciado permite a
opção sintética desde que tenha **≥ 1.000 linhas e ≥ 10 colunas** — este dataset
tem **1.200 linhas e 14 colunas**.

O gerador é **determinístico** (seed fixa) e **calibrado em estatísticas reais**:

* **Falcon 1** (2006–2009) com baixa confiabilidade (~40% de sucesso);
* **Falcon 9 / Falcon Heavy** altamente confiáveis (> 95%);
* missões **GTO** mais pesadas e levemente mais arriscadas;
* **reúso** de boosters crescente ao longo dos anos (≈ 0 antes de 2017);
* **sucesso de pouso** melhorando com o tempo (alvo alternativo mais balanceado).

Regenere com:

```bash
python scripts/generate_dataset.py 1200
```

> ⚠️ **Transparência:** os números do README foram obtidos com o dataset
> sintético de fallback. O caminho de ingestão real (API v4) é o oficial da
> entrega e produz um CSV com o mesmo schema; basta rodar `make ingest` seguido
> de `make train` para reexecutar tudo sobre dados reais.

## Schema (`spacex_launches.csv`)

| coluna             | tipo    | origem (API)                              | descrição                                  |
|--------------------|---------|-------------------------------------------|--------------------------------------------|
| `flight_number`    | int     | `launch.flight_number`                    | número sequencial do voo                   |
| `date_utc`         | str     | `launch.date_utc`                         | timestamp ISO-8601 do lançamento           |
| `year`             | int     | derivado de `date_utc`                    | ano do lançamento                          |
| `rocket`           | str     | `rockets[launch.rocket].name`             | versão do foguete                          |
| `payload_mass_kg`  | float   | soma de `payloads[*].mass_kg`             | massa total do payload (kg); pode ser nulo |
| `orbit`            | str     | `payloads[principal].orbit`               | órbita alvo (LEO, GTO, SSO, ISS, …)        |
| `launch_site`      | str     | `launchpads[launch.launchpad].name`       | local de lançamento                        |
| `reused`           | bool    | `cores[principal].reused`                 | booster reutilizado                        |
| `flights`          | int     | `cores[principal].flights`                | nº de voos acumulados do core              |
| `gridfins`         | bool    | `cores[principal].gridfins`               | presença de grid fins                      |
| `legs`             | bool    | `cores[principal].legs`                   | presença de pernas de pouso                |
| `landing_success`  | bool    | `cores[principal].landing_success`        | **alvo alternativo** (recuperação do 1º estágio) |
| `success`          | bool    | `launch.success`                          | **alvo principal** (sucesso do lançamento) |
| `upcoming`         | bool    | `launch.upcoming`                         | lançamento futuro (removido na limpeza)    |

### Convenções de resolução

* **Falcon Heavy tem 3 cores** → usamos o **primeiro core (central/primário)**
  para derivar `reused`, `flights`, `gridfins`, `legs`, `landing_success`.
* **Vários payloads** → a massa é **somada** e usamos a **órbita do payload
  principal** (primeiro com órbita definida).
* **Limpeza** (em [`features/cleaning.py`](../src/launch_success/features/cleaning.py)):
  remove `upcoming == True`, descarta linhas com o **alvo** nulo, e coage tipos
  (booleanos → `{0,1,NaN}`, numéricos → `float`). A imputação de `payload_mass_kg`
  (mediana) ocorre **dentro do pipeline**, só no treino, evitando *data leakage*.
