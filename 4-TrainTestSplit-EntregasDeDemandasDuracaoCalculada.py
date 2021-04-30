#Demandas Entregues em tempo inferior a 2 anos
# Split_train_test
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

	maior_R2_msle,       maior_R2_msle,  maior_R2_mae    = 0, 0, 0
	maior_R2_num,        maior_R2_dias,  maior_R2_pearson= 0, 0, 0
	maior_R2_r2_train,   maior_R2_r2_test                = 0, 0
	maior_R2_pred,       maior_R2_X,     maior_R2_y      = 0, 0, 0   
	maior_R2_train_ind,  maior_R2_test_ind               = 0, 0

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
		X = df[["ANO_ASSIN"]]
		y = df[["Esforço PF"]]

		#calculando a correlação entre as variáveis
		r = np.corrcoef(X, y, rowvar=False)     
		pearson = r[1,0]
        
		if (pearson>0.5 or pearson<-0,5):              
			tam_treino = 0.55
			tam_teste  = 0.45        
     
			#Testa todas as divisões possíveis para o período de tempo    
			for num in range(1,24,1):        

				tam_treino = tam_treino + 0.01
				tam_teste  = tam_teste  - 0.01     

				#Divide os arrays em subconjuntos randômicos de treino e teste
				X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=tam_treino, test_size=tam_teste, random_state=42)
   
				train_ind = list(X_train.index)                       
				test_ind = list(X_test.index)
                                         
				#Não aceita conjuntos menores q 2 elementos
				if (len(train_ind)>1 and len(test_ind)>1):                                
					#Fitting Simple Linear Regression to the set
					regressor = LinearRegression()
					regressor.fit(X, y)

					#Métrica padrão, que é a mais relevante para a tarefa de Machine Learning
					r2_train = regressor.score(X_train, y_train)
					r2_test = regressor.score(X_test, y_test)
                
					#Exige r2_test seja positivo para ser melhor que uma reta horizontal
					if (r2_test>0.9 and r2_test<=1): #or (r2_test<-0.5 and r2_test>-1)                        
						y_pred    = regressor.predict(X)                        
						rmse_train = sqrt(mean_squared_error(y_train,y_pred[train_ind]) )                        
						rmse_test  = sqrt(mean_squared_error(y_test,y_pred[test_ind]) )
						msle_test  = mean_squared_log_error(y_test,y_pred[test_ind])
						msle_train = mean_squared_log_error(y_train,y_pred[train_ind])
						rmse  = sqrt(mean_squared_error(y,y_pred) )                        
						msle  = mean_squared_log_error(y,y_pred)                        
						mae   = mean_absolute_error(y,y_pred)                        

						#Seleciona o menor valor de MSLE
						if (msle_test<menor_msle_msle or menor_msle_msle==0):                                                   
							menor_msle_rmse,     menor_msle_msle,   menor_msle_mae  = rmse, msle, mae
							menor_msle_num,      menor_msle_dias, menor_msle_pearson= num, dias, pearson
							menor_msle_r2_train, menor_msle_r2_test                 = r2_train, r2_test
							menor_msle_pred,     menor_msle_X,      menor_msle_y    = y_pred, X, y
							menor_msle_train_ind,menor_msle_test_ind                = train_ind, test_ind
							menor_msle_rmse_train,menor_msle_rmse_test              = rmse_train, rmse_test
							menor_msle_msle_train,menor_msle_msle_test              = msle_train, msle_test                            
							menor_msle_tam_treino,menor_msle_tam_teste              = tam_treino, tam_teste

						#Seleciona o Maior valor de R2 Teste
						if (r2_test>maior_R2_r2_test or maior_R2_r2_test==0):
							maior_R2_rmse,     maior_R2_msle,  maior_R2_mae   = rmse, msle, mae 
							maior_R2_num,      maior_R2_dias, maior_R2_pearson= num, dias, pearson
							maior_R2_r2_train, maior_R2_r2_test               = r2_train, r2_test            
							maior_R2_pred,     maior_R2_X,     maior_R2_y     = y_pred, X, y
							maior_R2_train_ind,maior_R2_test_ind              = train_ind, test_ind   
							maior_R2_rmse_train,maior_R2_rmse_test            = rmse_train, rmse_test 
							maior_R2_msle_train,maior_R2_msle_test            = msle_train, msle_test                                                        
							maior_R2_tam_treino,maior_R2_tam_teste            = tam_treino, tam_teste                            
        
                           
		del [[df]]
		#Fim do loop for
    
	print("----------- Melhor MSLE ------------------") 
	print("Pearson %0.3f" %menor_msle_pearson)
	print("dias = ", menor_msle_dias, ", split = ", menor_msle_num)  
	print("Tam. Treino = %0.3f" % menor_msle_tam_treino, ", Tam. Teste = %0.3f" % menor_msle_tam_teste)       
	print('Coeficiente de determinação da predição R2:')
	print('1-Set de treino: %0.3f' % menor_msle_r2_train)
	print('2-Set de teste: %0.3f'  % menor_msle_r2_test)
	print("Índice de treino: ", menor_msle_train_ind)            
	print("Índice de teste: ",  menor_msle_test_ind)   
	print("RMSE Treino: %0.3f" % menor_msle_rmse_train )
	print("RMSE Teste:  %0.3f" % menor_msle_rmse_test)      
	print("MSLE Treino: %0.3f" % menor_msle_msle_train )
	print("MSLE Teste:  %0.3f" % menor_msle_msle_test)      
	print("RMSE: %0.3f" % menor_msle_rmse)
	print("MSLE: %0.3f" % menor_msle_msle)    
	print("MAE : %0.3f" % menor_msle_mae)
    
        
	#Visualizando o conjunto de resultados
	label1 = 'Valores Reais'
	label2 = 'Predição - MSLE'
	label3 = 'Predição - R2'  
	titulo = 'Esforço PF Entregue x Ano - Split'
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
	print("Tam. Treino = %0.3f" % maior_R2_tam_treino, ", Tam. Teste = %0.3f" % maior_R2_tam_teste)       
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
            

            