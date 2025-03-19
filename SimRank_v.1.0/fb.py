import numpy as np
import csv
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import sys
import time
import unidecode
import matplotlib.pyplot as plt 
import datetime as dt
from matplotlib.ticker import FixedLocator, FixedFormatter
from memory_profiler import profile
import dlyap
import dlyap_fast

class G_operator:
	def __init__(self, A, c):
		self.A = A
		self.n = self.A.shape[1]
		if (self.n!=self.A.shape[0]):
			print(f"Warning! Non-zero adjacency matrix detected when constructing operator.")
		self.c = c
	def __call__(self, u):
		self.A = csr_matrix(self.A)
		U = u.reshape((self.n, self.n), order = 'F')
		ATUA = self.A.T@U@self.A
		G = U - self.c*ATUA+self.c*np.diag(np.diag(ATUA))
		G = G.reshape((self.n**2,1), order = 'F')
		return G

#WARNING: GetMatrix only for non-directed graphs.
def GetMatrix(n, path = "/home/egor-berezin/mticiansred_library/Uni_MGU-Sarov/Disser_mag/programs/SimRank_mainbranch/SNAP_datasets/facebook_combined.txt", delimeter = ' '):
	A = np.zeros((n,n))
	file = open(path,'r')
	for line in file:
		string = file.readline().split(delimeter)
		node1, node2 = int(string[0]), int(string[1])
		A[node1,node2] = 1
	A = (A+A.T) #Omit this for directed graphs!
	print("Edges: ", sum(sum(A)))
	return A

def norm1_ColumnNormalize(M): #may be optimized! L1 col norms can be easily obtained by sum(A).
	col_1_norms = np.sum(np.abs(M), axis = 0)
	col_1_norms[col_1_norms == 0] = 1 #Avoid div by 0
	print("Columns 1-norms:")
	print(col_1_norms)
	normalized = M/col_1_norms
	print("Column 1-normalized matrix:")
	print (normalized)
	return normalized

def Facebook(acc, m_Krylov):
	n = 4039
	A = GetMatrix(n)
	I = np.identity(n) #identity matrix of required dimensions
	print("Adjacency matrix:")
	print(A)
	c = 0.8
	A_n1c = norm1_ColumnNormalize(A)
	I = np.identity(n)
	print("Initializing GMRES...")
	k_iter = 1000000
	I_vec = np.identity(n).reshape((n**2,1), order = 'F')
	A_n1c_csr = csr_matrix(A_n1c)
	G = G_operator(A_n1c, c) 

	print(f"Starting GMRES with {k_iter} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
	
	S_end = dlyap.GMRES_m(G, m_Krylov, I_vec, I_vec, k_iter, acc).reshape((n,n), order = 'F') #reshape vec to mat
	
	#print("Starting MinRes...")
	#dlyap.MinRes(k_iter, A_n1c_csr, c, N, acc)
	#print("Starting Simple Iter...")
	#S_SI = dlyap.SimpleIterIP(k_iter, A_n1c_csr, c, 0.2, N, acc)
	#S_SI = dlyap.SimpleIterIP(k_iter, A_n1c_csr, c, 1, N, acc)
	'''
	print("Singular values::")
	sigma = np.linalg.svd(S_end, full_matrices = False)[1]
	print(sigma.tolist())
	'''
	plt.savefig("Facebook_eps_"+str(acc)+"_"+str(dt.datetime.now())+".png")
	print("Similarity matrix:")
	print(S_end)
	
	plt.figure()
	graph = plt.imshow(np.log(S_end-I+1e-15))
	cbar = plt.colorbar()
	cbar.set_label("ln(S[i,j])")
	#plt.title("Facebook")
	plt.title("Матрица S", fontweight = "bold")


def showmatrix():
	n = 4039
	A = GetMatrix(n)
	plt.figure()
	graph = plt.imshow(A+1e-15)
	return 0
