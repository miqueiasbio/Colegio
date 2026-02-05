# Sistema de IA para análise de preços de supermercado (Bahia)

Projeto em Python 3.10+ para:
1. detectar anomalias de preço,
2. prever preços de curto prazo (7 dias),
3. gerar relatórios e gráficos por município.

## Estrutura

```text
Colegio/
├── configs/
│   └── config.json
├── data/
│   ├── README.md
│   └── precos_exemplo.csv
├── models/
├── reports/
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── features.py
│   ├── model.py
│   ├── predict.py
│   ├── report.py
│   ├── train.py
│   └── utils.py
├── tests/
│   └── test_data_prep.py
└── requirements.txt
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dados esperados

CSV com colunas:
- `data` (YYYY-MM-DD)
- `municipio`
- `supermercado`
- `bairro` (opcional)
- `categoria`
- `produto`
- `marca` (opcional)
- `unidade`
- `quantidade`
- `preco`
- `promocao` (opcional, 0/1)

O pipeline faz:
- normalização de texto (acentos/maiúsculas/espaços),
- validação de schema,
- parsing de datas,
- limpeza de valores inválidos.

## Engenharia de atributos

- `preco_normalizado = preco / quantidade`
- média móvel 7 e 30 dias por produto/supermercado
- variação percentual vs 7 dias atrás
- sazonalidade (dia da semana, mês)
- codificação categórica (`supermercado`, `categoria`, `produto`)

## Modelos

### 1) Previsão de preços
- Modelo: `ForecastLSTM` (PyTorch)
- Entrada: janelas temporais (`window_size`, padrão 30)
- Saída: próximos 7 dias (`forecast_horizon`)

### 2) Detecção de anomalias
- Modelo: `PriceAutoencoder` (PyTorch)
- Score: erro de reconstrução
- Limiar: percentil configurável (`anomaly_percentile`)

Também é calculado baseline por média móvel para comparação.

## CLI

Treinar:
```bash
python -m src.train --config configs/config.json
```

Avaliar:
```bash
python -m src.evaluate --config configs/config.json
```

Prever produto específico:
```bash
python -m src.predict --config configs/config.json --produto "arroz tipo 1 5kg"
```

Gerar relatórios:
```bash
python -m src.report --config configs/config.json
```

## Saídas esperadas em `reports/`

- `previsoes_<municipio>.csv`
- `anomalias_<municipio>.csv`
- `variacao_item_supermercado_<municipio>.csv`
- `ranking_supermercados_<municipio>.csv`
- `serie_real_vs_prevista_<produto>.png`
- `boxplot_supermercado_<municipio>.png`
- `ranking_cesta_<municipio>.png`

## Observações

- O filtro geográfico é por `municipio` via configuração.
- O código é genérico e funciona para qualquer município com dados suficientes.
- Não utiliza serviços externos; roda totalmente local.
