#Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Importando o dataset
location = "extracao_sgpti_03.csv"

try:
	print ("Lendo arquivo ", location, "...")
	df = pd.read_csv(location,encoding='iso-8859-1',delimiter =';', quotechar='"', thousands='.', decimal=',')

	#Determina intervalos para seleção
	df = df.loc[ df['ANO_ASSIN']<=2020]
	df = df.loc[ df['ANO_ASSIN']>=2012]

	#soma os valores em PF das demandas, agrupados por Ano
	df_sub_nivel = df.groupby(["ANO_ASSIN"]).sum().reset_index()

	#calculando a correlação entre as variáveis
	print("Coeficiente de Correlação de Pearson:")
	print(df_sub_nivel.corr(method='pearson'))

	# seleciona as colunas "Ano" e "Esforço PF"
	X = df_sub_nivel[["ANO_ASSIN"]]
	y = df_sub_nivel[["Esforço PF"]]

	#Divide os arrays em subconjuntos randômicos de treino e teste
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.30, test_size=0.30, random_state=42)

	#Fitting Simple Linear Regression to the set
	regressor = LinearRegression()
	regressor.fit(X, y)

	#Métrica padrão, que é a mais relevante para a tarefa de Machine Learning
	r2_train = regressor.score(X_train, y_train)
	r2_test = regressor.score(X_test, y_test)

	print('Coeficiente de determinação da predição R2:')
	print('1-Set de treino: %.2f' % r2_train)
	print('2-Set de teste: %.2f' % r2_test)

	#Visualizando o conjunto de resultados
	label1 = 'Valores Reais'
	label2 = 'Predição'
	titulo = 'Esforço PF Entregue x Ano - Split'
	eixo_x = 'Ano'
	eixo_y = 'Esforço PF'    
       
	plt.scatter(X, y, color = 'red', label=label1)
	plt.plot(X,  regressor.predict(X), color = 'blue', label=label2)
	plt.grid(which='major', linestyle='--')
	plt.minorticks_on()
	plt.title(titulo)
	plt.xlabel(eixo_x)
	plt.ylabel(eixo_y)
	plt.legend([label2, label1], loc=2)          
	plt.show()

except IOError:
    print ("Arquivo ", location, " não foi encontrado")
