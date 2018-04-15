#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
import numpy as np
import math

### Preprocessamento (retirado de BooleanModel.py)

	#Separa a entrada em tokens, utilizando como delimitadores ',', '.', '!', '?' e ' '
def tokenize(text):
	return [tokens.strip().replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split() for tokens in text]

#Realiza a remoção das stopwords informadas e entradas duplicadas e normaliza a entrada para caixa-baixa
def normalize(tokens, stopwords):
	norm = []
	for phrase in tokens:
		norm.append([token.lower() for token in phrase if token.lower() not in stopwords])

	return np.unique(np.hstack(np.array(norm)))
###

### Term-frequency e dicionário

#Calcula a frequência de cada token nos documentos e a frequencia dos documentos invertidas, retornando um dicionario com essas informações
# retorna td, idf dos termos 
def create_index( tokens, docs):
	#gera frequencia dos termos
	tf = {token : np.char.count([doc.lower() for doc in docs], token) for token in tokens}
	#gera matriz de frequencia invertida dos documentos
	idf = {token: tf[token].nonzero()[0].size for token in tokens}

	return tf, idf

#Calcula os pesos conforme tf e idf
# retorna matriz dos pesos de cada termo
def gen_tf_idf(tf,idf,n_docs):
	weight_matrix = {}
	for token in tf:
		#inicia indice pelo termo
		weight_matrix[token] = []
		for doc in range(n_docs):
			#verifica se o termo aparece no respectivo doc
			if(tf[token][doc] > 0):
				# calcula o peso
				tf_idf = (1 + math.log(tf[token][doc], 2)) * math.log(n_docs/idf[token], 2)
			else:
				tf_idf = 0
			weight_matrix[token].append(tf_idf)

	return weight_matrix
###

if __name__ == '__main__':

	text = ['O peã e o caval são pec de xadrez. O caval é o melhor do jog.', 'A jog envolv a torr, o peã e o rei.', 'O peã lac o boi', 'Caval de rodei!', 'Polic o jog no xadrez.']
	stopwords = ['a', 'o', 'e', 'é', 'de', 'do', 'no', 'são']
	q = ['xadrez peã caval torr']

	tokens = normalize(tokenize(text), stopwords)
	tf, idf = create_index(tokens, text)
	weighted_tokens = gen_tf_idf(tf, idf, len(text))

	for token in tokens:
		print str(token) + " |"
		for doc in text:
			print str(text.index(doc)) + ": " + str(weighted_tokens[token][text.index(doc)])
