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
	df = df.loc[ df['ANO_ASSIN']>=2005]

	#soma os valores em PF das demandas, agrupados por Ano
	df = df[["ANO_ASSIN",'Ano','Esforço PF']].groupby(by=['ANO_ASSIN','Ano']).count().reset_index()
    
	X = df["ANO_ASSIN"]
	y = df["Ano"]
	z = df["Esforço PF"]    
    
	#Visualizando o conjunto de resultados
	area = (3*z)
	color= (z)
	plt.scatter(X, y, s=area, c=color, alpha=0.5)
	#plt.scatter(X, y, color = 'red')
	#plt.plot(X, regressor.predict(X), color = 'blue')

	plt.grid(which='major', linestyle='--')
	plt.minorticks_on()
	plt.title('Num.Demandas: Ano Início x Ano Término')
	plt.xlabel('Conclusão Demanda')
	plt.ylabel('Abertura Demanda')
    
	plt.show()
except IOError:
    print ("Arquivo ", location, " não foi encontrado")
