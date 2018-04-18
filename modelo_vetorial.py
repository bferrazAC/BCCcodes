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

def calc_vect(tokens, n_docs):
	docs_norms = np.array([0 for n in range(n_docs)])
	# lista de vetores dos docs
	docs_vect = []

	for n in range(n_docs):
		# vetor singular do documento
		doc_vect = []
		for token in tokens:
			w = tokens[token][n]
			doc_vect.append(w)
			docs_norms[n] += w*w
		docs_vect.append(doc_vect)

	if(n_docs == 1):
		docs_vect = docs_vect[0]

	return docs_vect,docs_norms

def rank(tokens, q_tokens, n_docs):
	# rank_docs = [(n,0) for n in n_docs]

	docs_vect, docs_norms = calc_vect(tokens,n_docs)
	q_vect, q_norms = calc_vect(q_tokens, 1)

	for doc in range(n_docs):
		print docs_vect[doc]
		print q_vect
		print "llalalal"
		r = np.dot(docs_vect[doc],q_vect)
		r /= math.sqrt(docs_norms[doc]*q_norms)
		rank_docs[doc] = (r, doc)

	rank_docs.sort()

	return rank_docs


###

if __name__ == '__main__':

	text = ['O peã e o caval são pec de xadrez. O caval é o melhor do jog.', 'A jog envolv a torr, o peã e o rei.', 'O peã lac o boi', 'Caval de rodei!', 'Polic o jog no xadrez.']
	stopwords = ['a', 'o', 'e', 'é', 'de', 'do', 'no', 'são']
	q = ['xadrez peã caval torr']
	# Trata os termos dos docs
	tokens = normalize(tokenize(text), stopwords)
	tf, idf = create_index(tokens, text)
	weighted_tokens = gen_tf_idf(tf, idf, len(text))

	# Trata os termos da consulta
	q_tokens = normalize(tokenize(q), stopwords)
	q_tf, q_idf = create_index(q_tokens, q)
	q_weighted_tokens = gen_tf_idf(q_tf, q_idf, len(q))

	rank_docs = rank(weighted_tokens,q_weighted_tokens,len(text))

	for token in tokens:
		print str(token) + " |"
		for doc in text:
			print str(text.index(doc)) + ": " + str(weighted_tokens[token][text.index(doc)])


	print rank_docs