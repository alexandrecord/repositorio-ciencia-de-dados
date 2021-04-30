from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Implementação de funções
def PlotaGrafico(X,y, cor, pred, cor_pred, titulo, eixo_x, eixo_y):
	#Visualizando o conjunto de resultados
	plt.scatter(X, y, color = cor)
	plt.plot(X, pred, color = cor_pred)
	plt.grid(which='major', linestyle='--')
	plt.minorticks_on()
	plt.title(titulo)
	plt.xlabel(eixo_x)
	plt.ylabel(eixo_y)
	plt.show()



y_true = [3, 5, 6.5, 9.5, 10.4, 13.5, 14.5, 17.5, 18.4, 20.6]
X_true = [[0,1], [1,2],[2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10]]
X = [1,2,3,4,5,6,7,8,9,10]

#Divide os arrays em subconjuntos randômicos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_true, y_true, train_size=0.30, test_size=0.30, random_state=42)
   
#Fitting Simple Linear Regression to the set
regressor = LinearRegression()
regressor.fit(X_true, y_true)

#Métrica padrão, que é a mais relevante para a tarefa de Machine Learning
print('r2_train = %0.3f' %regressor.score(X_train, y_train))
print('r2_test = %0.3f'  %regressor.score(X_test, y_test))
                
y_pred    = regressor.predict(X_true)  

print('MSE: %0.3f' %mean_squared_error(y_true, y_pred))
print('RMSE:%0.3f' %sqrt(mean_squared_error(y_true, y_pred)) )    
print('MSLE:%0.3f' %mean_squared_log_error(y_true, y_pred))
print('MAE: %0.3f' %mean_absolute_error(y_true, y_pred))

PlotaGrafico(X,y_true, 'blue', y_pred, 'red', 'Métricas', 'X', 'y')

#--------------------- tudo multiplicado por 1.000
print('------- Agora com todos os dados multiplicados por 1.000 --------')
y_true = [3000, 5000, 6500, 9500, 10400, 13500, 14500, 17500, 18400, 20600]

#Divide os arrays em subconjuntos randômicos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_true, y_true, train_size=0.30, test_size=0.30, random_state=42)
   
#Fitting Simple Linear Regression to the set
regressor = LinearRegression()
regressor.fit(X_true, y_true)

#Métrica padrão, que é a mais relevante para a tarefa de Machine Learning
print('r2_train = %0.3f' %regressor.score(X_train, y_train))
print('r2_test = %0.3f'  %regressor.score(X_test, y_test))
                
y_pred    = regressor.predict(X_true)   

print('MSE: %0.3f' %mean_squared_error(y_true, y_pred))
print('RMSE:%0.3f' %sqrt(mean_squared_error(y_true, y_pred)) )    
print('MSLE:%0.3f' %mean_squared_log_error(y_true, y_pred))
print('MAE: %0.3f' %mean_absolute_error(y_true, y_pred))

PlotaGrafico(X,y_true, 'blue', y_pred, 'red', 'Métricas', 'X','y*1000', )