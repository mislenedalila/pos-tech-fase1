import pandas as pd
import matplotlib.pyplot as plt
# para deixar todas as saídas com os mesmos valores obtidos na live.
import numpy as np
np.random.seed(42)
import os

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



dataset = pd.read_csv("housing.csv")

#aula referência: https://github.com/FIAP/CURSO_IA_ML/blob/main/Aula%202/Case%20end%20to%20end%20Regress%C3%A3o.ipynb

#print(dataset.head())

# primeiro passo para analisar dados nulos de cada coluna do arquivo
# conseguimos identicar o tipo de cada coluna, na ocean_proximity é do tipo object
#dataset.info()

#com esse comeando é possível verificar as colunas que tem nesse campo ocean_proximity
#print(set(dataset["ocean_proximity"]))

#contar quantas categorias possuem
#print(dataset["ocean_proximity"].value_counts())


#para analisar dataset do tipo númerico pode usar comando describe, que retorna algumas informações estatisticas 
#print(dataset.describe())

#para visualar os graficos de histograma, tem um termo chamado outliner que são dados fora da curva
#para o nosso desafio para gerar os gráficos podemos utilizar esse comeando
#dataset.hist(bins=50, figsize=(20,15))
#plt.show()


#separando as bases de treino e teste
#from sklearn.model_selection import train_test_split

#teste_size estou falando que estou usando 20% da minha base para teste e 80% para treinamento
#df_train, df_test = train_test_split(dataset, test_size = 0.2, random_state = 7)

#mostra a quantidade de dados que será utilizada para treinamento e testes
#print(len(df_train), "treinamento +", len(df_test), "teste")

# Divida por 1,5 para limitar o número de categorias de renda
# dividindo o valor da coluna "median_income" de cada entrada pelo valor 1,5 e, em seguida, arredondando o resultado para cima usando a função
# np.ceil() (da biblioteca NumPy). Isso cria uma nova coluna chamada "income_cat" no dataset que contém os valores das categorias de renda após
# a divisão e arredondamento.


df_train, df_test = train_test_split(dataset, test_size = 0.2, random_state = 7)

#print(len(df_train), "treinamento +", len(df_test), "teste")

dataset["median_income"].hist()

dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5) # ceil para arredondar valores para cima


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")

plt.show()