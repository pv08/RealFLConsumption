import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

add_md("# Análise de Resultados: Baseline vs TimeVAE\n\nEste notebook consolida a avaliação dos modelos de Baseline (treinados com dados reais) e TimeVAE (TSTR - treinados com dados sintéticos), além de checar a qualidade visual e a estabilidade de treino.")

add_md("## 1. Importações e Configurações")
add_code("""import os
import glob
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Configurações visuais do seaborn
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)""")

add_md("## 2. Desempenho Preditivo (Baseline vs TSTR)\n\nVamos ler os CSVs das duas abordagens e juntar em um único DataFrame para comparar os erros.")
add_code("""def carregar_resultados():
    frames = []
    
    # Lendo Baseline
    baseline_files = glob.glob("etc_baseline/TimeVAE/austin/results/*/*_BASELINE_BASELINE_metrics_*.csv")
    for f in baseline_files:
        df = pd.read_csv(f)
        df['Metodologia'] = 'Baseline'
        frames.append(df)
        
    # Lendo TimeVAE (TSTR)
    tstr_files = glob.glob("etc_timevae/TimeVAE/austin/results/*/*_TSTR_A6_metrics_*.csv")
    for f in tstr_files:
        df = pd.read_csv(f)
        df['Metodologia'] = 'TimeVAE (TSTR)'
        frames.append(df)
        
    if not frames:
        print("Nenhum arquivo CSV encontrado! Verifique se as pastas estão no local correto.")
        return pd.DataFrame()
        
    return pd.concat(frames, ignore_index=True)

df_resultados = carregar_resultados()
df_resultados.head()""")

add_code("""# Vamos plotar o MSE e o MAE divididos por Modelo (LSTM, GRU, RNN) e coloridos pela Metodologia
if not df_resultados.empty:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vamos usar as colunas reais reportadas no dataframe ('inv_MSE', 'inv_MAE')
    metric_mse = 'inv_MSE' if 'inv_MSE' in df_resultados.columns else 'MSE'
    metric_mae = 'inv_MAE' if 'inv_MAE' in df_resultados.columns else 'MAE'
    
    sns.boxplot(data=df_resultados, x='model', y=metric_mse, hue='Metodologia', ax=axes[0])
    axes[0].set_title(f"Comparação de {metric_mse.upper()}")
    
    sns.boxplot(data=df_resultados, x='model', y=metric_mae, hue='Metodologia', ax=axes[1])
    axes[1].set_title(f"Comparação de {metric_mae.upper()}")
    
    plt.tight_layout()
    plt.show()""")

add_md("## 3. Qualidade e Fidelidade Visual (Distribuição Latente)\n\nAqui, visualizamos rapidamente algumas das imagens t-SNE geradas pelo TimeVAE para garantir que os dados sintéticos estão acompanhando a distribuição dos reais.")
add_code("""# Lista algumas imagens t-SNE (de teste e treino) para um CID qualquer
plot_files = glob.glob("etc_timevae/TimeVAE/austin/results/plots/*_tsne_*.png")
plot_files = sorted(plot_files)

if plot_files:
    print(f"Exibindo 2 de {len(plot_files)} plots encontrados:")
    # Exibe as 2 primeiras apenas como exemplo
    for f in plot_files[:2]:
        print(f)
        display(Image(filename=f))
else:
    print("Nenhuma imagem t-SNE encontrada em results/plots/")""")

add_md("## 4. Estabilidade e Eficiência de Treinamento (TimeVAE)\n\nVamos extrair as perdas de treinamento do gerador a partir dos arquivos `.pkl` para ver como o modelo convergiu.")
add_code("""log_files = glob.glob("etc_timevae/TimeVAE/austin/logs/*-train_val.pkl")
log_files = sorted(log_files)

if log_files:
    print(f"Encontrados {len(log_files)} arquivos de log. Plotando o primeiro como exemplo:")
    # Pega apenas o primeiro log para amostra
    sample_log = log_files[0]
    
    with open(sample_log, 'rb') as f:
        log_data = pickle.load(f)
        
    if isinstance(log_data, dict):
        plt.figure(figsize=(10, 5))
        if 'train_total_loss' in log_data:
            plt.plot(log_data['train_total_loss'], label='Treino Total Loss')
        if 'val_total_loss' in log_data:
            plt.plot(log_data['val_total_loss'], label='Validação Total Loss')
            
        plt.title(f"Curvas de Treino/Validação: {os.path.basename(sample_log)}")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        print("O formato do log não é um dicionário simples. Verifique os dados armazenados:", type(log_data))
else:
    print("Nenhum log .pkl encontrado!")""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("analise_resultados.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
    
print("Notebook analise_resultados.ipynb gerado com sucesso!")
