#Esforço x Tempo Entrega
import matplotlib.pyplot as plt
import pandas as pd

#Importando o dataset
location = "extracao_sgpti_03.csv"

df = pd.read_csv(location,encoding='iso-8859-1',delimiter =';', quotechar='"', thousands='.', decimal=',')

#Determina intervalos para seleção
df = df.loc[ df['ANO_ASSIN']<=2020]
df = df.loc[ df['ANO_ASSIN']>=2012]
       
#soma os valores em PF das demandas, agrupados por Ano
df = df.groupby(["TEMPO_ENTREGA","Esforço PF"]).count().reset_index()
df = df[['TEMPO_ENTREGA','Esforço PF','ANO_ASSIN']].groupby(by=['TEMPO_ENTREGA','Esforço PF']).count().reset_index()

# seleciona as colunas "Ano" e "Esforço PF"
X = df[["TEMPO_ENTREGA"]]
y = df[["Esforço PF"]]
z = df["ANO_ASSIN"]  
    
#Visualizando o conjunto de resultados
color= (z**2)
area = (z**2)
plt.scatter(X, y, s=area, c=color, alpha=0.5)


#Visualizando o conjunto de resultados
#plt.scatter(X, y, color = 'blue')
plt.grid(which='major', linestyle='--')
plt.minorticks_on()
plt.title('Esfoço x Tempo Entrega')
plt.xlabel('Tempo Entrega')
plt.ylabel('Esforço')
plt.show()

