#Demandas Entregues em tempo inferior a 2 anos
#Método de criação de subconjuntos -> KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from math import sqrt


#Importando o dataset
location = "extracao_sgpti_03.csv"

try:
	menor_msle_rmse,     menor_msle_msle,   menor_msle_mae    = 0, 0, 0
	menor_msle_num,      menor_msle_dias,   menor_msle_pearson= 0, 0, 0
	menor_msle_r2_train, menor_msle_r2_test                   = 0, 0
	menor_msle_pred,     menor_msle_X,      menor_msle_y      = 0, 0, 0
	menor_msle_train_ind,menor_msle_test_ind                  = 0, 0
	menor_msle_msle_train,menor_msle_msle_test                = 0, 0

	maior_R2_rmse,       maior_R2_msle,  maior_R2_mae    = 0, 0, 0
	maior_R2_num,        maior_R2_dias,  maior_R2_pearson= 0, 0, 0
	maior_R2_r2_train,   maior_R2_r2_test                = 0, 0
	maior_R2_pred,       maior_R2_X,     maior_R2_y      = 0, 0, 0   
	maior_R2_train_ind,  maior_R2_test_ind               = 0, 0
	maior_R2_msle_train,maior_R2_msle_test               = 0, 0
    

	print ("Lendo arquivo ", location, "...")    
    
	#Testa várias durações de demandas
	for dias in range(20,3648,5):        
		df = pd.read_csv(location,encoding='iso-8859-1',delimiter =';', quotechar='"', thousands='.', decimal=',')

		#Determina intervalos para seleção
		df = df.loc[ df['ANO_ASSIN']<=2020]
		df = df.loc[ df['ANO_ASSIN']>=2012]        
        
		#seleciona demandas com tempo de construção inferior a 1 ano
		df = df.loc[ df['TEMPO_ENTREGA']<dias]    

		#soma os valores em PF das demandas, agrupados por Ano
		df = df.groupby(["ANO_ASSIN"]).sum().reset_index()

		# seleciona as colunas "Ano" e "Esforço PF"
		X = np.array(df[["ANO_ASSIN"]])
		y = np.array(df[["Esforço PF"]])

		#calculando a correlação entre as variáveis
		r = np.corrcoef(X, y, rowvar=False)
		pearson = r[1,0]
        
		#Não trabalhar com o módulo do coef. de pearson abaixo de 0,5 para não trabalhar com relação linear fraca.            
		if (pearson>0.5 or pearson<-0,5):
            
			#Testa todas as divisões possíveis para o período de tempo    
			for num in range(2,9,1):    
                
				kf = KFold(n_splits=num)
				for train_index, test_index in kf.split(X):
                    
					#Não aceita conjuntos menores q 2 elementes
					if (len(train_index)>1 and len(test_index)>1):
                        
						X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]         

						#Fitting Simple Linear Regression to the set
						regressor = LinearRegression()
						regressor.fit(X, y)

						#Métrica padrão, que é a mais relevante para a tarefa de Machine Learning
						r2_train = regressor.score(X_train, y_train)
						r2_test = regressor.score(X_test, y_test)

						#Exige r2_test seja positivo para ser melhor que uma reta horizontal
						if (r2_test>0.9 and r2_test<=1): #or (r2_test<-0.5 and r2_test>-1)
							y_pred    = regressor.predict(X)                        
							rmse_train = sqrt(mean_squared_error(y_train,y_pred[train_index]) )                        
							rmse_test  = sqrt(mean_squared_error(y_test,y_pred[test_index]) )
							msle_test  = mean_squared_log_error(y_test ,y_pred[test_index])
							msle_train = mean_squared_log_error(y_train,y_pred[train_index])   
							#print("msle treino %0.2f" % msle_train, "msle teste %0.2f" % msle_test)
							rmse  = sqrt(mean_squared_error(y,y_pred) )                        
							msle  = mean_squared_log_error(y,y_pred)                        
							mae   = mean_absolute_error(y,y_pred)                        

							#Seleciona o menor valor de RMSE
							if (msle_test<menor_msle_msle or menor_msle_msle==0):                                                    
								menor_msle_rmse,     menor_msle_msle,   menor_msle_mae  = rmse, msle, mae
								menor_msle_num,      menor_msle_dias, menor_msle_pearson= num, dias, pearson
								menor_msle_r2_train, menor_msle_r2_test                 = r2_train, r2_test
								menor_msle_pred,     menor_msle_X,      menor_msle_y    = y_pred, X, y
								menor_msle_train_ind,menor_msle_test_ind                = train_index, test_index
								menor_msle_msle_train,menor_msle_msle_test              = msle_train, msle_test                                 
								menor_msle_rmse_train,menor_msle_rmse_test              = rmse_train, rmse_test

							#Seleciona o Maior valor de R2 Teste
							if (r2_test>maior_R2_r2_test or maior_R2_r2_test==0):
								maior_R2_rmse,     maior_R2_msle,  maior_R2_mae   = rmse, msle, mae 
								maior_R2_num,      maior_R2_dias, maior_R2_pearson= num, dias, pearson
								maior_R2_r2_train, maior_R2_r2_test               = r2_train, r2_test            
								maior_R2_pred,     maior_R2_X,     maior_R2_y     = y_pred, X, y
								maior_R2_train_ind,maior_R2_test_ind              = train_index, test_index   
								maior_R2_msle_train,maior_R2_msle_test            = msle_train, msle_test                                 
								maior_R2_rmse_train,maior_R2_rmse_test            = rmse_train, rmse_test                                                        
                            
		del [[df]]
	#Fim do loop for
    
	print("----------- Melhor RMSE ------------------") 
	print("Pearson %0.3f" %menor_msle_pearson)
	print("dias = ", menor_msle_dias, ", split = ", menor_msle_num)                    
	print('Coeficiente de determinação da predição R2:')
	print('1-Set de treino: %0.3f' % menor_msle_r2_train)
	print('2-Set de teste: %0.3f'  % menor_msle_r2_test)
	print("Índice de treino: ", menor_msle_train_ind)            
	print("Índice de teste: ",  menor_msle_test_ind)   
	print("RMSE Treino: %0.3f" % menor_msle_msle_train )
	print("RMSE Teste:  %0.3f" % menor_msle_rmse_test)   
	print("MSLE Treino: %0.3f" % menor_msle_msle_train)
	print("MSLE Teste:  %0.3f" % menor_msle_msle_test)          
	print("RMSE: %0.3f" % menor_msle_rmse)
	print("MSLE: %0.3f" % menor_msle_msle)    
	print("MAE : %0.3f" % menor_msle_mae)
    
        
	#Visualizando o conjunto de resultados
	label1 = 'Valores Reais'
	label2 = 'Predição - MSLE'
	label3 = 'Predição - R2'  
	titulo = 'Esforço PF Entregue x Ano - KFold'
	eixo_x = 'Ano'
	eixo_y = 'Esforço PF'    
    
	plt.scatter(menor_msle_X, menor_msle_y, color = 'red', label=label1)
	plt.plot(menor_msle_X, menor_msle_pred, color = 'blue', label=label2)
	plt.grid(which='major', linestyle='--')
	plt.minorticks_on()
	plt.title(titulo)
	plt.xlabel(eixo_x)
	plt.ylabel(eixo_y)
	plt.legend([label2, label1], loc=1)          
	plt.show()

    
	print("----------- Melhor R2 ------------------")  
	print("Pearson %0.3f" %maior_R2_pearson)    
	print("dias = ", maior_R2_dias, ", split = ", maior_R2_num)                    
	print('Coeficiente de determinação da predição R2:')   
	print('1-Set de treino: %0.3f' % maior_R2_r2_train)
	print('2-Set de teste: %0.3f'  % maior_R2_r2_test)
	print("Índice de treino: ", maior_R2_train_ind)            
	print("Índice de teste: ",  maior_R2_test_ind)                        
	print("RMSE Treino: %0.3f" % maior_R2_rmse_train )
	print("RMSE Teste:  %0.3f" % maior_R2_rmse_test)  
	print("MSLE Treino: %0.3f" % maior_R2_msle_train )
	print("MSLE Teste:  %0.3f" % maior_R2_msle_test)              
	print("RMSE: %0.3f" % maior_R2_rmse)
	print("MSLE: %0.3f" % maior_R2_msle)    
	print("MAE : %0.3f" % maior_R2_mae)
    
        
	#Visualizando o conjunto de resultados
	plt.scatter(maior_R2_X, maior_R2_y, color = 'red', label=label1)
	plt.plot(maior_R2_X, maior_R2_pred, color = 'blue', label=label3)
	plt.grid(which='major', linestyle='--')
	plt.minorticks_on()
	plt.title(titulo)
	plt.xlabel(eixo_x)
	plt.ylabel(eixo_y)
	plt.legend([label3, label1], loc=1)          
	plt.show()
            
except IOError:
    print ("Arquivo ", location, " não foi encontrado")
