import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.model_selection import train_test_split

#teste_size estou falando que estou usando 20% da minha base para teste e 80% para treinamento
df_train, df_test = train_test_split(dataset, test_size = 0.2, random_state = 7)

#mostra a quantidade de dados que será utilizada para treinamento e testes
print(len(df_train), "treinamento +", len(df_test), "teste")