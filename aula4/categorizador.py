from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Conjunto de dados completo com exemplos ampliados
textos = [
    # Tecnologia
    "O novo lançamento da Apple",
    "Atualização no mundo da tecnologia",
    "Apple lança novo iPhone",
    "Google anuncia nova versão do Android",
    "Inovação em inteligência artificial",
    "Microsoft lança nova atualização do Windows",
    "Samsung revela tecnologia de telas dobráveis",

    # Esportes
    "Resultado do jogo de ontem",
    "Campeonato de futebol",
    "Seleção brasileira vence amistoso",
    "Jogador bate recorde no campeonato",
    "Final do torneio de vôlei feminino",
    "Resultados da Fórmula 1 do fim de semana",
    "Equipe conquista medalha de ouro nas Olimpíadas",

    # Política
    "Eleições presidenciais",
    "Política internacional",
    "Nova reforma tributária é aprovada",
    "Discurso presidencial causa polêmica",
    "Eleições municipais serão no próximo mês",
    "Senado debate projeto de lei sobre saúde",
    "Cúpula internacional discute mudanças climáticas",
    "Presidente veta projeto de lei aprovado pelo Congresso",
    "Nova legislação sobre proteção de dados",
    "Política econômica será debatida amanhã",
    "Congresso analisa proposta de orçamento",
    "Governo anuncia medidas contra inflação",
    "Prefeito inaugura nova escola municipal",
    "Discussões sobre política ambiental ganham força",
    "Parlamentares discutem sobre segurança pública",
    "Ministro da educação anuncia mudanças nas universidades",
    "Justiça eleitoral prepara eleições nacionais"
]

categorias = [
    "tecnologia", "tecnologia", "tecnologia", "tecnologia", "tecnologia", "tecnologia", "tecnologia",
    "esportes", "esportes", "esportes", "esportes", "esportes", "esportes", "esportes",
    "política", "política", "política", "política", "política", "política", "política",
    "política", "política", "política", "política", "política", "política", "política",
    "política", "política", "política" 
]

# Treinamento do modelo
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)
clf = MultinomialNB()
clf.fit(X, categorias)

# Função para classificar novas frases
def classificar_frase(frase):
    frase_transformada = vectorizer.transform([frase])
    return clf.predict(frase_transformada)[0]

# Classificando novas frases
novas_frases = [
    "Governo anuncia nova lei",
    "Final da Copa Libertadores",
    "Samsung lança novo celular",
    "Tecnologia e inovação no futebol"
]

resultados = pd.DataFrame({
    "Frase": novas_frases,
    "Categoria Predita": [classificar_frase(frase) for frase in novas_frases]
})

print(resultados)
