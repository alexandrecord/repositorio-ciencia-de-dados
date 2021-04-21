#Importação da biblioteca Pandas
import pandas as pd

origem  = 'extracao_sgpti_datas_01.csv'
destino = 'extracao_sgpti_03.csv'

print ("Arquivo ",origem, " não foi encontrado")
try:
	print ("Lendo arquivo ", origem, "...")
	#Leitura do arquivo CSV e criação do DataFrame
	df = pd.read_csv(origem,encoding='iso-8859-1',delimiter =';', quotechar='"', thousands='.', decimal=',')

	#Converte as datas que estavam com o tipo string para o tipo datetime
	df['Data de Aprovação do Titular'] = pd.to_datetime(df['Data de Aprovação do Titular'], format='%d/%m/%Y')
	df['Data Assinatura Anexo 4/5 Prest. Serv.'] = pd.to_datetime(df['Data Assinatura Anexo 4/5 Prest. Serv.'], format='%d/%m/%Y')

	#Cria duas novas colunas: 
	# 1) ANO_ASSI, contendo somente o ano da entrega da demanda;
	# 2) TEMPO_ENTREGA, contendo o número, em dias, 
	df['ANO_ASSIN']     = df['Data Assinatura Anexo 4/5 Prest. Serv.'].dt.year
	df['TEMPO_ENTREGA'] = (df['Data Assinatura Anexo 4/5 Prest. Serv.']-df['Data de Aprovação do Titular']).dt.days

	#seleciona os registros os quais possuem assinatura de entrega do prestador de serviço
	df = df.loc[df['Data Assinatura Anexo 4/5 Prest. Serv.'].notnull()]
	df = df.loc[df['Data de Aprovação do Titular'].notnull()]

	try:
		df.to_csv(destino,encoding='iso-8859-1', index=False, sep=';', decimal=',')
	except IOError:
	    print ("Erro ao criar arquivo: ",destino)
	finally:
	        print ("Arquivo: ",destino, "gerado!")

except IOError:
    print ("Arquivo ", origem, "não foi encontrado")