import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = df['overweight'] = (df['weight'] / (df['height'] ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].replace({1: 0, 2: 1, 3: 1})
df['gluc'] = df['gluc'].replace({1: 0, 2: 1, 3: 1})

# 4
def draw_cat_plot():
    # Transformar os dados em formato "melt"
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Agrupar e reformular os dados para mostrar a contagem de cada característica
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Criar todas as combinações possíveis de 'cardio', 'variable', e 'value'
    all_combinations = pd.MultiIndex.from_product(
        [df_cat['cardio'].unique(), df_cat['variable'].unique(), df_cat['value'].unique()],
        names=['cardio', 'variable', 'value']
    )

    # Redefinir os dados para incluir todas as combinações possíveis, preenchendo com 0 se estiver faltando
    df_cat = df_cat.set_index(['cardio', 'variable', 'value']).reindex(all_combinations, fill_value=0).reset_index()

    # Desenhar o gráfico categórico
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')

    return fig.fig

# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_hi'] >= df['ap_lo']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # Gerar uma máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurar a figura
    fig, ax = plt.subplots(figsize=(12, 8))

    # Desenhar o mapa de calor
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=.5, ax=ax, cmap='coolwarm')

    return fig