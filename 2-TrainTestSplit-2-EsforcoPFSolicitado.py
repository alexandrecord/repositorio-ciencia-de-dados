#Importando as bibliotecas
#Demandas Solicitadas
#Método de criação de subconjuntos -> train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from math import sqrt


#Importando o dataset
location= "extracao_sgpti_01.csv"

try:
	menor_rmse_rmse_train,menor_rmse_rmse_test, menor_rmse_num  = 0, 0, 0
	menor_rmse_r2_train,  menor_rmse_r2_test                    = 0, 0
	menor_rmse_train_ind, menor_rmse_test_ind,menor_rmse_pearson= 0, 0, 0
	menor_rmse_pred,      menor_rmse_X,       menor_rmse_y      = 0, 0, 0
	maior_rmse_tam_treino,maior_rmse_tam_teste                  = 0, 0
	menor_rmse_rmse, menor_rmse_msle, menor_rmse_mae            = 0, 0, 0
    
	maior_R2_rmse_train, maior_R2_rmse_test, maior_R2_num  = 0, 0, 0
	maior_R2_r2_train,   maior_R2_r2_test                  = 0, 0
	maior_R2_train_ind, maior_R2_test_ind,maior_R2_pearson = 0, 0, 0
	maior_R2_pred,      maior_R2_X,       maior_R2_y       = 0, 0, 0   
	maior_R2_tam_treino,maior_R2_tam_teste                 = 0, 0 
    
	print ("Lendo arquivo ", location, "...")
	df = pd.read_csv(location,encoding='iso-8859-1',delimiter =';', quotechar='"', thousands='.', decimal=',')

	#Determina intervalos para seleção
	df = df.loc[ df['Ano']<=2020]
	df = df.loc[ df['Ano']>=2012]

	#soma os valores em PF das demandas, agrupados por Ano
	df = df.groupby(["Ano"]).sum().reset_index()

	# seleciona as colunas "Ano" e "Esforço PF"
	X = df[["Ano"]]
	y = df[["Esforço PF"]]

	#calculando a correlação entre as variáveis
	r = np.corrcoef(X, y, rowvar=False)     
	pearson = r[1,0]  
    
	#Não trabalhar com o módulo do coef. de pearson abaixo de 0,5 para não trabalhar com relação linear fraca.
	if (pearson>0.5 or pearson<-0,5):    
        
		#testar diferentes proporções entre os tamanhos de teste e treino        
		tam_treino = 0.55
		tam_teste  = 0.45        
    
		#Testa todas as divisões possíveis para o período de tempo    
		for num in range(1,24,1): #9       
			tam_treino = tam_treino + 0.01
			tam_teste  = tam_teste  - 0.01        
			#Divide os arrays em subconjuntos randômicos de treino e teste
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=tam_treino, test_size=tam_teste, random_state=42)      
            
			#Fitting Simple Linear Regression to the set
			regressor = LinearRegression()
			regressor.fit(X, y)

			#Métrica padrão, que é a mais relevante para a tarefa de Machine Learning
			r2_train = regressor.score(X_train, y_train)
			r2_test  = regressor.score(X_test, y_test)    

			#Exige r2_test seja positivo para ser melhor que uma reta horizontal
			if (r2_test>0.9 and r2_test<=1):         
				y_pred    = regressor.predict(X) 
				train_ind = list(X_train.index)                       
				test_ind = list(X_test.index)

				rmse_train = sqrt(mean_squared_error(y_train,y_pred[train_ind]) )                        
				rmse_test  = sqrt(mean_squared_error(y_test,y_pred[test_ind]) )
				rmse  = sqrt(mean_squared_error(y,y_pred) )                        
				msle  = mean_squared_log_error(y,y_pred)                        
				mae   = mean_absolute_error(y,y_pred)                        
            

				#Seleciona o menor valor de RMSE no teste
				if (rmse_test< menor_rmse_rmse or menor_rmse_rmse==0):                              
					menor_rmse_rmse_train,menor_rmse_rmse_test            = rmse_train, rmse_test, 
					menor_rmse_num,       menor_rmse_pearson              = num, pearson                            
					menor_rmse_r2_train,  menor_rmse_r2_test              = r2_train, r2_test
					menor_rmse_train_ind, menor_rmse_test_ind             = train_ind, test_ind, 
					menor_rmse_pred,      menor_rmse_X,       menor_rmse_y= y_pred, X, y
					menor_rmse_tam_treino,menor_rmse_tam_teste            = tam_treino, tam_teste                
					menor_rmse_rmse,      menor_rmse_msle, menor_rmse_mae = rmse, msle, mae             

				#Seleciona o Maior valor de R2 Teste
				if (r2_test>maior_R2_rmse_test or maior_R2_rmse_test==0):
					maior_R2_rmse_train, maior_R2_rmse_test, maior_R2_num  = rmse_train, rmse_test, num
					maior_R2_r2_train,   maior_R2_r2_test                  = r2_train, r2_test
					maior_R2_train_ind, maior_R2_test_ind,maior_R2_pearson = train_ind, test_ind, pearson                     
					maior_R2_pred,      maior_R2_X,       maior_R2_y       = y_pred, X, y
					maior_R2_tam_treino,maior_R2_tam_teste                 = tam_treino, tam_teste
					maior_R2_rmse,      maior_R2_msle,    maior_R2_mae     = rmse, msle, mae                              
                            
		del [[df]]
	#Fim do loop for
    
	print("----------- Melhor RMSE ------------------") 
	print("Pearson = %0.3f" %menor_rmse_pearson)
	print("Tam. Treino = %0.3f" % menor_rmse_tam_treino,
          ", Tam. Teste = %0.3f" % menor_rmse_tam_teste)                                       
	print('Coeficiente de determinação da predição R2:')
	print('1-Set de treino: %0.3f' % menor_rmse_r2_train)
	print('2-Set de teste: %0.3f'  % menor_rmse_r2_test)
	print("Índice de treino: ", menor_rmse_train_ind)            
	print("Índice de teste: ",  menor_rmse_test_ind)                        
	print("RMSE Treino: %0.3f" % menor_rmse_rmse_train )
	print("RMSE Test: %0.3f"   % menor_rmse_rmse_test)    
	print("RMSE: %0.3f" % menor_rmse_rmse)
	print("MSLE: %0.3f" % menor_rmse_msle)    
	print("MAE : %0.3f" % menor_rmse_mae)
    
        
	#Visualizando o conjunto de resultados
	label1 = 'Valores Reais'
	label2 = 'Predição - RMSE'
	label3 = 'Predição - R2'  
	titulo = 'Esforço PF Solicitado x Ano - Split'
	eixo_x = 'Ano'
	eixo_y = 'Esforço PF'
	plt.scatter(menor_rmse_X, menor_rmse_y, color = 'green', label=label2)
	plt.plot(menor_rmse_X, menor_rmse_pred, color = 'blue', label=label1)
	plt.grid(which='major', linestyle='--')
	plt.minorticks_on()
	plt.title(titulo)
	plt.xlabel(eixo_x)
	plt.ylabel(eixo_y)       
	plt.legend([label2, label1], loc=1)    
	plt.show()

    
	print("----------- Melhor R2 ------------------")  
	print("Pearson = %0.3f" %maior_R2_pearson)
	print("Tam. Treino = %0.3f" % maior_R2_tam_treino, ", Tam. Teste = %0.3f" % maior_R2_tam_teste)                                       
	print('Coeficiente de determinação da predição R2:')
	print('1-Set de treino: %0.3f' % maior_R2_r2_train)
	print('2-Set de teste: %0.3f'  % maior_R2_r2_test)
	print("Índice de treino: ", maior_R2_train_ind)            
	print("Índice de teste: ",  maior_R2_test_ind)                        
	print("RMSE Treino: %0.3f" % maior_R2_rmse_train )
	print("RMSE Teste:  %0.3f" % maior_R2_rmse_test)    
	print("RMSE: %0.3f" % maior_R2_rmse)
	print("MSLE: %0.3f" % maior_R2_msle)    
	print("MAE : %0.3f" % maior_R2_mae)
    
        
	#Visualizando o conjunto de resultados
	plt.scatter(maior_R2_X, maior_R2_y, color = 'green', label=label3)
	plt.plot(maior_R2_X, maior_R2_pred, color = 'blue', label=label1)
	plt.grid(which='major', linestyle='--')
	plt.minorticks_on()
	plt.title(titulo)
	plt.xlabel(eixo_x)
	plt.ylabel(eixo_y)
	plt.legend([label3, label1], loc=1)    
	plt.show()
    
except IOError:
    print ("Arquivo ", location, " não foi encontrado")
