# Case de Prevenção à Fraudes @ Mercado Livre

Welcome to FREEDOM, a leader in the online commerce and payment systems market in your region, with over 25 years of history and a successful growth track record.

At FREEDOM, Fraud Prevention is the reference sector internally and externally on all security topics for transactions within its ecosystem. It ensures the security of users' operations through the application of tools, predictive-analytical models, and innovative technologies. It is one of the company's areas that allows taking risks and accelerating growth.
As a new member of the FREEDOM Fraud Prevention team, you have been allocated to your first data science project, congratulations!
For this project, the Analytics team has already extracted a sample of transactions that have been labeled as Fraud (fraud column with a value equal to 1) or Non-Fraud (fraud column with a value equal to 0). They also included the scores from an existing fraud predictive model in production (score column).

**Your main objectives in this project are:**
1. *Establish the baseline*: for this, it is necessary to evaluate the performance of the current machine learning model that is in production.
    - What is the predictive performance of the model? Which metrics are most appropriate for this analysis and why?
2. *Train new machine learning model(s) for fraud prediction*: those are going to be assessed as candidates to replace the current one.
   - You are free to generate new features if appropriate, and include any technique or analysis you believe suits the case. 
    - Specify the techniques and algorithms used for both: data preprocessing stage, and and to train the new model(s)? Comment on your decisions.
    - Explicitly compare the new model(s) you trained with the current model in terms of predictive performance?
3. *Business metrics and decision making based on models*: define a cutoff point to reject transactions based on the output of the model
Consider that FREEDOM's commission rate is 5% on the value of a correctly approved payment (monto column), and for each approved fraudulent transaction we lose 100% of the payment value. The cutoff point should maximize FREEDOM's profit based on this definition.

**Deliverables**:
- Jupyter notebook properly commented and submitted.
Requisites:
The notebook must be self contained, including the solutions, visualizations, and decisions made during the resolution of the case. 
Generate appropriate figures/graphs, use markdown annotations, and explain the necessary inferences. 
Any person should be able to read the notebook and understand each of the development stages and the reasoning behind them. You will be also evaluated based on the clarity, effective visualizations, and story telling of your analysis and modeling process.

**Files**:
- `meli_fraud_prevention_case.ipynb`: Python notebook containing the case description and information on how to install packages and read the dataset to be used.
- `fraud_dataset_v2.csv`: accessible at https://limewire.com/d/c4a2438f-b41d-40ee-955a-6d0b180b1a2d#mPBNISG8nbrYtdavEhExOXhEPrT0ls8QaOwfFkjyaG8
  
**Other instructions:**
- Feel free to create and use other files (folders, notebooks, scripts, etc.).
- When you are done, create a zip file with all your resources (all files required to run and understand your solution) and send us in reply to your email.

Last, but not least: have fun :) 

# Resolução

## Introdução
Para resolução do case utilizarei o framework [cookiecutter-datas-science](https://cookiecutter-data-science.drivendata.org/) com algumas adaptações. O framework tem como objetivo facilitar o desenvolvimento separando as etapas de treinamento do modelo em scripts e notebooks garantindo replicabilidade e fácil experimentação.  
Ao final do desenvolvimento vou reunir os resultados em um único notebook, para facilitar a entrega.

## 1.Replicabilidade
Para garantir a replicabilidade dos experimentos, crie um ambiente virtual python idêntico ao utilizado no projeto.

```bash
pip install -r requirements.txt
```

## 2.Avaliação do Baseline
Começar avaliando o modelo já existente na tabela.

[2-Analyse-Baseline.ipynb](notebooks/2-Analyse-Baseline.ipynb)


# Modelagem
Após avaliação do modelo vamos propor a construção de um novo modelo, tentando superar ou então capturar diferentes padrões do modelo existente.  

### 3.Separação Tabela
Etapa para separação da tabela entre treino, validação e teste.  
Para treino será utilizado 80% do período [20200308, 20200407].  
Para validação será utilizado 20% restante do período de treino [20200308, 20200407].  
Para teste será utilizado 100% do período [20200408, 20200421].  

Essa etapa pode ser executada via notebook ou script.  

[3-Split-Date.ipynb](notebooks/3-Split-Data.ipynb)

```bash
python src/data/split_data.py --dataset_name=fraud_dataset_v2 --ymd_train=20200308 --ymd_test=20200408
```

### 4.Processamento Básico
Etapa de preparação dos dados, verificando nulos e tipos.  
Foram corrigidos os dados abaixo:  
Ajustes de Tipo:  
- Ajustar coluna de data 'fecha' para tipo timestamp  

Ajuste de Nulos:  
- Ajustar colunas ['d', 'g', 'o', 'q'], para as colunas categóricas vou criar uma nova categoria indicando o nulo. 
- Ajustar colunas ['b', 'c', 'f', 'l', 'm'], para as colunas numéricas vou realizar uma substituição utilizando a mediana, para isso vou utilizar o conjunto de validação. 
Colunas com até 70 valores únicos foram definidas como colunas do tipo categórico e colunas com mais foram definidas como numéricas.  

Essa etapa pode ser executada via notebook ou script.  

[4-Data-Basic-Process.ipynb](notebooks/4-Data-Prep.ipynb)

```bash
python src/data/basic_process.py --dataset_name=fraud_dataset_v2
```

### 5.Feature Engineering
Criação de novas variáveis baseada nas existentes no dataset.  
Como não tenho informações sobre o que é cada uma das variáveis, foram criadas apenas algumas baseadas na data.  

Essa etapa pode ser executada via notebook ou script.  

[5-Feature-Engineering.ipynb](notebooks/5-Feature-Engineering.ipynb)

```bash
python src/features/create_features.py --dataset_prefix=fraud_dataset_v2
```

### 6.Análise Exploratória
Após a criação das variáveis, vamos realizar uma EDA para entender sobre a relação entre as variáveis explicativas e a variável resposta, assim como a relação entre as variáveis explicativas.  

Essa etapa é totalmente executada via notebook.  

[6-EDA.ipynb](notebooks/6-EDA.ipynb)

### 7.Feature Store
Após a análise exploratória tive algumas ideias para criação de novas variáveis, então nessa etapa serão criadas variáveis que não se encontram no dataset/payload e que podem ser construídas a partir de estruturas de feature stores.  

Nessa etapa será criado 1 *feature groups*:  
- Perfil Categoria Produto últimos 7 dias

A ideia desse feature group é trazer informações históricas dos produtos condensados.  

Essa etapa pode ser executada via notebook ou script.  

[7-Feature-Store.ipynb](notebooks/7-Feature-Store.ipynb)

```bash
python src/features/create_feature_stores.py --dataset_prefix=fraud_dataset_v2
```

E para enriquecer as variáveis nas tabelas:

```bash
python python src/features/enrich_feature_stores.py --dataset_prefix=fraud_dataset_v2
```

### 8.Feature Selection
Mesmo com um problema com poucas variáveis vamos realizar um método de seleção de features.  
Vamos utilizar como forma de seleção o algoritmo RandomForest com os parametros a seguir:
- `n_estimators=100` : Quantidade razoaável de árvores  
- `criterion='gini'` : Eficiência computacional vs entropia  
- `max_depth=5` : Quantidade razoável de nós  
- `max_features='sqrt` : Oportunidade de interação de diferentes features  
- `class_weight='balanced'` : Problema desbalanceado, dando maior peso para classes minoritárias  
A escolha do algoritmo se deve ao fato de ser uma forma simples de avaliar a importância de cada uma das variáveis, avaliando um método de árvore (mesmo que será utilizado no algoritmo final) e selecioanando as variáveis, dando oportunidade para elas aparecerem em diferentes árvores interagindo com diferentes variáveis.  

No meio das variáveis serão colocadas 4 variáveis aleatórias, 2 categóricas (baixa cardinalidade) e 2 continuas.  
As variáveis selecionadas serão aquelas que se mantiverem acima da primeira aleatória no ranking de importância por `ganho de informação`.  

(método parecido com Boruta, porém computacionalmente mais rápido)

Essa etapa pode ser executada via notebook ou script.  


### 9.Hupertunning


### 10.Treinamento


### 11.Avaliação


### 12.Análise de Erro


## Project Organization

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------