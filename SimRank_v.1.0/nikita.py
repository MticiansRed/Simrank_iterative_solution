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
import dlyap_nikita as dlyap
import dlyap_fast

#WARNING: GetMatrix only for non-directed graphs.
def GetMatrix(N):
	A = np.array(
	[[1, 0, -0.01, 0.01],
	[0, 1, 500000, -500000],
	[232558139.53, 0, 9292.32, -9292.32],
	[0, 17407407407.47, 500000000, 1]])
	print(A)
	return A
	
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

def SolveDLyap(acc, m_Krylov, ip, N):
	A = GetMatrix(N)
	k_iter = 1000000
	print(f"{k_iter} iterations")
	print("Starting GMRES...")
	S_end = dlyap.GMRES_m(k_iter, m_Krylov, A, N, acc)
	#print("Starting MinRes...")
	#dlyap.MinRes(k_iter, A_n1c_csr, c, N, acc)
	#print("Starting Simple Iter...")
	#dlyap.SimpleIterIP(k_iter, A_n1c_csr, c, ip, N, acc)
	'''
	print("Singular values::")
	sigma = np.linalg.svd(S_end, full_matrices = False)[1]
	print(sigma.tolist())
	'''
	#plt.savefig("Artificial_eps_"+str(acc)+"_"+str(dt.datetime.now())+".png")
	print("Solution:")
	print(S_end)
	
	plt.figure()
	#graph = plt.imshow(np.log(S_end-I+1e-15)) #logaritmic
	graph = plt.imshow(S_end)
	cbar = plt.colorbar()
	cbar.set_label("ln(S[i,j])")
	#plt.title("Facebook")
	plt.title("Матрица S", fontweight = "bold")


def showmatrix():
	N = 4039
	A = GetMatrix(N)
	plt.figure()
	graph = plt.imshow(A+1e-15)
	return 0

if __name__=="__main__":
	
	plt.figure()
	plt.grid()
	#metro.Metro(1e-5, 15)
	#fb.Facebook(1e-5, 15)
	SolveDLyap(20, 15, 0.1, 4)
	#fb.showmatrix()
	plt.show()
	

