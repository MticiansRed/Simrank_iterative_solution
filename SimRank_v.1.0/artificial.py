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

global A_
global N_
global c_

#WARNING: GetMatrix only for non-directed graphs.
def GetMatrix(N):
	A = np.random.randint(low=0, high=2, size=(N,N)) #demo matrix
	print("Random matrix:")
	print(A)
	return A

def generate_random_matrix(N, density=0.01):
    """
    Generates an NxN matrix with elements 0 or 1.
    
    Parameters:
        N (int): Size of the matrix (NxN).
        density (float): Probability of a cell being 1 (between 0 and 1).
        
    Returns:
        np.ndarray: NxN matrix with random 0s and 1s.
    """
    if not (0 <= density <= 1):
        raise ValueError("Density must be between 0 and 1.")
    
    # Create a random matrix and apply the density
    random_matrix = np.random.rand(N, N) < density
    return random_matrix.astype(int)

def norm1_ColumnNormalize(M): #may be optimized! L1 col norms can be easily obtained by sum(A).
	col_1_norms = np.sum(np.abs(M), axis = 0)
	col_1_norms[col_1_norms == 0] = 1 #Avoid div by 0
	print("Columns 1-norms:")
	print(col_1_norms)
	normalized = M/col_1_norms
	print("Column 1-normalized matrix:")
	print (normalized)
	return normalized

def SolveDLyap(acc, m_Krylov, ip, N):
	A = GetMatrix(N)
	I = np.identity(N) #identity matrix of required dimensions
	print("Adjacency matrix:")
	print(A)
	c = 0.96
	A_n1c = norm1_ColumnNormalize(A) 
	I = np.identity(N)
	print("Similarity matrix:")
	k_iter = 1000000
	print(f"{k_iter} iterations")
	A_n1c_csr = csr_matrix(A_n1c)
	
	A_ = A_n1c_csr
	N_ = N
	c_ = c
	print("Starting GMRES...")
	#S_end = dlyap.GMRES_m(k_iter, m_Krylov, A_n1c_csr, c, N, acc)
	print("Starting GMRES scipy..")
	S_end = dlyap.GMRES_scipy(k_iter, m_Krylov, A_n1c_csr, c, N, acc)
	#print("Starting MinRes...")
	#dlyap.MinRes(k_iter, A_n1c_csr, c, N, acc)
	print("Starting Simple Iter...")
	#dlyap.SimpleIterIP(k_iter, A_n1c_csr, c, ip, N, acc)
	S_end = dlyap.SimpleIterIP(k_iter, A_n1c_csr, c, 1, N, acc)
	'''
	print("Singular values::")
	sigma = np.linalg.svd(S_end, full_matrices = False)[1]
	print(sigma.tolist())
	'''
	#plt.savefig("Artificial_eps_"+str(acc)+"_"+str(dt.datetime.now())+".png")
	print("Similarity matrix:")
	print(S_end)
	
	plt.figure()
	#graph = plt.imshow(np.log(S_end-I+1e-15)) #logaritmic
	graph = plt.imshow(S_end-I)
	cbar = plt.colorbar()
	cbar.set_label("ln(S[i,j])")
	#plt.title("Facebook")
	plt.title("Матрица S", fontweight = "bold")
	topn = 3
	for count in range(topn): #top n
		S_end_triu = np.triu(S_end)
		smax = np.argmax(S_end_triu-I, axis=None)
		indmax = np.unravel_index(smax, (S_end-I).shape)
		print("Maximum SimScore: ", S_end[indmax])
		print("Maximum SimScore index: ", indmax)
		S_end[indmax] = 0

def showmatrix():
	N = 4039
	A = GetMatrix(N)
	plt.figure()
	graph = plt.imshow(A+1e-15)
	return 0
