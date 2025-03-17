import numpy as np
import csv
import pandas as pd
import sys
import time
import unidecode
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt 
import networkx as nx
from matplotlib.ticker import FixedLocator, FixedFormatter
from memory_profiler import profile
import dlyap

#---Functions

def IsIn(j, M): #j = '12', nbs = '112 123'
	res = 0
	M = unidecode.unidecode(M)
	M_lst = M.split(' ')
	#print(M_lst)
	M_int = [int(a) for a in M_lst]
	for a in M_int:
		if (a == int(j)):
			res = 1
	return res

def L1_ColumnNormalize(M):
	col_l1_norms = np.sum(np.abs(M), axis = 0)
	col_l1_norms[col_l1_norms == 0] = 1 #Avoid div by 0
	print("Columns L1-norms:")
	print(col_l1_norms)
	normalized = M/col_l1_norms
	print("Column L1-normalized matrix:")
	print (normalized)
	return normalized
#--- Main procedures

def Metro(acc, m_Krylov, path = "/home/egor-berezin/mticiansred_library/Uni_MGU-Sarov/Disser_mag/programs/SimRank_mainbranch/SimRank_v.1.0/data/metro_stations.csv" ):
	N = 303
	data = pd.read_csv(path)
	M_ind = data['Station_index']
	M_name = data['English_name']
	M_cpy = M_ind.copy()
	M_nbs = data['Line_Neighbors']
	M_trs = data['Transfers']
	M_trs = M_trs.fillna('0') #filling NaNs with '0'
	G = np.zeros((N,N))

	for i in M_ind:
		for j in M_cpy:
			if ( IsIn(str(j) , M_nbs[i-1]) or IsIn(str(j) , M_trs[i-1]) ): #Making connection if connected or transfers
				G[i-1][j-1] = 1 #i,j or j,i?
	
	print("Trace-check before nullifiyng:")
	print(np.trace(G))
	
	for i in range(len(G)):
		if (G[i,i] > 0):
			G[i,i] = 0 #Nullifying self-transfers: self-transfers must be omitted.
	
	print("Trace-check after nullifiyng:")
	print(np.trace(G))

	I = np.identity(N) #identity matrix of required dimensions
	A = np.zeros(N) #adjacency matrix
	A = G
	print("Adjacency matrix:")
	print(A)
	c = 0.8
	A_n1c = L1_ColumnNormalize(A) 
	I = np.identity(N)
	print("Similarity matrix:")
	k_iter = 1000000
	print("Starting GMRES...")
	S_end = dlyap.GMRES_m(k_iter, m_Krylov, A_n1c, c, N, acc)
	print(S_end)
	
	plt.savefig("Metro_eps_"+str(acc)+"_"+str(dt.datetime.now())+".png")
	print("Similarity matrix:")
	print(S_end)
	
	plt.figure()
	graph = plt.imshow(S_end-I)
	plt.title("Metro")
	plt.title("Metro", fontweight = "bold")
	'''
	G = nx.from_numpy_matrix(np.matrix(S_end-I), create_using=nx.DiGraph)
	layout = nx.spring_layout(G)
	nx.draw(G, layout)
	labels = nx.get_edge_attributes(G, "weight")
	nx.draw_networkx_edge_labels(G, pos=layout, edge_labels = labels)
	'''
	'''
	#plt.figure()
	
	fig, ax = plt.subplots()
	im = ax.imshow(S_end)

	# Show all ticks and label them with the respective list entries
	ax.set_xticks(np.arange(len(M_name)), labels=M_name)
	ax.set_yticks(np.arange(len(M_name)), labels=M_name)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(M_name)):
		for j in range(len(M_name)):
			text = ax.text(j, i, S_end[i, j],
						   ha="center", va="center", color="w")

	ax.set_title("Similarity matrix")
	fig.tight_layout()
	'''
	topn = 100
	for count in range(topn): #top n
		S_end_triu = np.triu(S_end)		
		smax = np.argmax(S_end_triu-I, axis=None)
		indmax = np.unravel_index(smax, (S_end-I).shape)
		print("Maximum SimScore: ", S_end[indmax])
		print("Maximum SimScore index: ", indmax)
		print(M_name[indmax[0]],'-', M_name[indmax[1]])
		S_end[indmax] = 0



