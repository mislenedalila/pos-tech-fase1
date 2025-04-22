import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Carregar os dados
df = pd.read_csv("insurance.csv")

# 2. Pré-processamento: codificar variáveis categóricas
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# 3. Definir variáveis independentes (X) e dependente (y)
X = df.drop("charges", axis=1)
y = df["charges"]

# 4. Dividir em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 6. Fazer previsões
y_pred = modelo.predict(X_test)

# 7. Calcular métricas
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 8. Exibir resultados
print("📊 Métricas do modelo:")
print(f"➡️ Mean Squared Error (MSE): {mse:.2f}")
print(f"➡️ Mean Absolute Error (MAE): {mae:.2f}")
print(f"➡️ R² Score: {r2:.4f}")
