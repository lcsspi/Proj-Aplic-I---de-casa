import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carregar o dataset
file_path = "https://raw.githubusercontent.com/lcsspi/Proj-Aplic-I-E-de-casa/refs/heads/main/Dataset/avocado.csv"
avocado_data = pd.read_csv(file_path)

# Converter a coluna 'Date' para o formato datetime
avocado_data['Date'] = pd.to_datetime(avocado_data['Date'])

# 1. Estatísticas descritivas
print("Estatísticas descritivas:")
print(avocado_data.describe())

# 2. Preço médio ao longo do tempo
plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='AveragePrice', data=avocado_data, ci=None)
plt.title('Preço Médio dos Abacates ao Longo do Tempo', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Preço Médio (USD)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Distribuição do preço médio
plt.figure(figsize=(10,6))
sns.histplot(avocado_data['AveragePrice'], bins=30, kde=True)
plt.title('Distribuição do Preço Médio', fontsize=16)
plt.xlabel('Preço Médio (USD)', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()
plt.show()

# 4. Preço médio por tipo de abacate (convencional vs orgânico)
plt.figure(figsize=(10,6))
sns.boxplot(x='type', y='AveragePrice', data=avocado_data)
plt.title('Comparação de Preço Médio por Tipo de Abacate', fontsize=16)
plt.xlabel('Tipo', fontsize=12)
plt.ylabel('Preço Médio (USD)', fontsize=12)
plt.tight_layout()
plt.show()

# 5. Volume total vendido por região
plt.figure(figsize=(12,8))
region_volume = avocado_data.groupby('region')['Total Volume'].sum().sort_values(ascending=False)
sns.barplot(x=region_volume.values, y=region_volume.index, palette="viridis")
plt.title('Volume Total Vendido por Região', fontsize=16)
plt.xlabel('Volume Total', fontsize=12)
plt.ylabel('Região', fontsize=12)
plt.tight_layout()
plt.show()

# 6. Tendência do preço médio por ano
plt.figure(figsize=(10,6))
avocado_data['year'] = avocado_data['Date'].dt.year
sns.lineplot(x='year', y='AveragePrice', hue='type', data=avocado_data, ci=None)
plt.title('Tendência do Preço Médio por Ano', fontsize=16)
plt.xlabel('Ano', fontsize=12)
plt.ylabel('Preço Médio (USD)', fontsize=12)
plt.tight_layout()
plt.show()

# 7. Volume total vendido por tipo de abacate
plt.figure(figsize=(10,6))
type_volume = avocado_data.groupby('type')['Total Volume'].sum()
sns.barplot(x=type_volume.index, y=type_volume.values, palette="coolwarm")
plt.title('Volume Total Vendido por Tipo de Abacate', fontsize=16)
plt.xlabel('Tipo', fontsize=12)
plt.ylabel('Volume Total', fontsize=12)
plt.tight_layout()
plt.show()

# 8. Correlação entre variáveis numéricas
numeric_data = avocado_data.select_dtypes(include=[np.number])  # Seleciona apenas colunas numéricas
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap de Correlações entre Variáveis Numéricas', fontsize=16)
plt.tight_layout()
plt.show()