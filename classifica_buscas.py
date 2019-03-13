from collections import Counter
import pandas as pd 

#testes multinomialNB:
#home, busca, logado
#home, busca
#home, logado
#busca, logado
#busca 82% em todos os casos
#testes ADABOSTCLASSIFIER
#home, busca, logado = 85%

df = pd.read_csv('buscas.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1


tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = int(porcentagem_de_teste * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

fim_de_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]

# A eficácia do algoritmo que chuta 0 ou 1
acerto_base = max(Counter(validacao_marcacoes).values())

taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)

print("Taxa de acerto base: ", taxa_de_acerto_base)

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    acertos = (resultado == teste_marcacoes)

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos
    print("Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto))
    return taxa_de_acerto

from sklearn.naive_bayes import MultinomialNB
modeloMultinomialNB = MultinomialNB()
resultadoMultinomialNB = fit_and_predict("MultinomialNB", modeloMultinomialNB, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBosstClassifier = AdaBoostClassifier()
resultadoAdaBosstClassifier = fit_and_predict("AdaBoostClassifier", modeloAdaBosstClassifier, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoAdaBosstClassifier > resultadoMultinomialNB:
    vencedor = modeloAdaBosstClassifier
else:
    vencedor = modeloMultinomialNB

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos

print("Taxa de acerto do melhor algoritmo: {0}".format(taxa_de_acerto))