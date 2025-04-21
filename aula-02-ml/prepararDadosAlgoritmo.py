import pandas as pd
import matplotlib.pyplot as plt
# para deixar todas as saídas com os mesmos valores obtidos na live.
import numpy as np
np.random.seed(42)
import os

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



dataset = pd.read_csv("housing.csv")

dataset["median_income"].hist()

dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5) # ceil para arredondar valores para cima


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1) # apagando a target para a base de treino (nosso x)
housing_labels = strat_train_set["median_house_value"].copy() #armazenando a target (nosso y)

# listando as colunas nulas

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

# contando as colunas nulas
print(housing.isnull().sum())

#tratando os valores nulos, pode substituir pela mediana
#Opção 1
# Substituindo os valores nulos pela mediana
median = housing["total_bedrooms"].median()
print(sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True))
print(sample_incomplete_rows)


#PCA para unificar colunas - redução de dimensionalidade 