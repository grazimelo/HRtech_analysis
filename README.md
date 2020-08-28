Dados de uma empresa de RH. 

Dados disponiveis em : https://www.kaggle.com/krismurphy01/data-lab

├── LICENSE
├── README.md          <- Descrição do Projeto
├── data
│   
│   ├── inter          <- Dados com processamento intermediário (modelo + interpretabilidade)
│   ├── processed      <- Dados com processamento finalizado (pós-pipeline de transformação)
│   └── raw            <- Dados originais utilizados na modelagem.
|
├── models             <- Arquivos binários (.sav, .pkl) com os modelos e logs de previsão em .txt
│
├── notebooks          <- Jupyter notebooks contendo o passo a passo da solução.
├── img                <- Imagens geradas na análise exploratória e demais etapas do projeto.
│
├── requirements.txt   <- Arquivo que contém todas as dependências do ambiente.
│
├── src                <- Pacote contendo todos os módulos, submódulos, funções e classes do projeto.
│   ├── __init__.py    <- Torna src um pacote.
│   ├── data_cleaning.py <- Módulo de limpeza de dados.
│   ├── selecao_de_features <- Módulo de seleção de features.
│   │── modelagem_metricas.py <- Módulo de modelagem e avaliação.
│
└── predict.py            <- Script que roda o modelo e realiza a previsão.
