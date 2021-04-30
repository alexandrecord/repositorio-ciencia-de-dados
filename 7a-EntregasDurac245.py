#Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Importando o dataset
location = "extracao_sgpti_03.csv"

try:
	tempo_entrega = 245
	print ("Lendo arquivo ", location, "...")
	df = pd.read_csv(location,encoding='iso-8859-1',delimiter =';', quotechar='"', thousands='.', decimal=',')
    
	#Determina intervalos para seleção
	df = df.loc[ df['ANO_ASSIN']<=2020]
	df = df.loc[ df['ANO_ASSIN']>=2005]
	df = df.loc[ df['TEMPO_ENTREGA']<tempo_entrega]      

	#soma os valores em PF das demandas, agrupados por Ano
	df = df[["ANO_ASSIN",'Ano','Esforço PF']].groupby(by=['ANO_ASSIN','Ano']).sum().reset_index()
    
	# seleciona as colunas "Ano", "ANO_ASSIM" e "Esforço PF"    
	X = df["ANO_ASSIN"]
	y = df["Ano"]
	z = df["Esforço PF"]    
    
	#Configura cor e tamanho das bolhas, com base no somatório de esforço em PF
	area = (z/25)
	color= (z/25)
    
	#Visualizando o conjunto de resultados 
	label1 = 'Tempo Entrega < %d' % tempo_entrega
	titulo = 'PF entregue: Ano Início x Ano Término'
	eixo_x = 'Conclusão Demanda'
	eixo_y = 'Abertura Demanda'
    
	plt.scatter(X, y, s=area, c=color, alpha=0.5, label=label1)
	plt.grid(which='major', linestyle='--')
	plt.title(titulo)
	plt.xlabel(eixo_x)
	plt.ylabel(eixo_y)
	plt.legend([label1], loc=2)      
	plt.show()
except IOError:
    print ("Arquivo ", location, " não foi encontrado")