import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import scipy.sparse as scsp
import sys
import time

global _A_
global _N_
global _c_


def GetRandomMatrix(N):
	A = np.random.randint(low=0, high=2, size=(N,N)) #demo matrix
	print("Random matrix:")
	print(A)
	return A

def GetMatrix(N, path = "/home/egor-berezin/mticiansred_library/Uni_MGU-Sarov/Disser_mag/programs/SimRank_mainbranch/SNAP_datasets/facebook_combined.txt", delimeter = ' '):
	A = np.zeros((N,N))
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

def InitGlobals(N, c, obtain_matrix):
	global _A_
	global _N_
	global _c_
	_A_ = norm1_ColumnNormalize(obtain_matrix(N))
	print(_A_)
	_N_ = N
	_c_ = c
	return 0


def G_matvec(u): #Liear operator function. G(S) = I, G(S) = S-c*A.T@S@A+c*diag(A.T@S@A). u comes as !vector! and then converted to matrix.
	global _A_
	global _N_
	global _c_
	_A_ = csr_matrix(_A_)
	U = u.reshape((_N_, _N_), order = 'F')
	ATUA = _A_.T@U@_A_
	G = U - _c_*ATUA+_c_*np.diag(np.diag(ATUA))
	G = G.reshape((_N_**2,1), order = 'F')
	return G

def my_callback(x):
	print('Current residual =', x)

def GMRES_scipy(c, N, path = "/home/egor-berezin/mticiansred_library/Uni_MGU-Sarov/Disser_mag/programs/SimRank_mainbranch/SNAP_datasets/facebook_combined.txt", delimeter = ' '):
	global _A_
	global _N_
	global _c_
	iterations = [0]
	residuals = [0]
	InitGlobals(N, c, GetMatrix)
	print(_A_)
	G = scsp.linalg.LinearOperator((N**2,N**2), matvec = G_matvec)
	I_vec = np.identity(N).reshape((N**2,1), order = 'F')
	st = time.time()
	s, data = scsp.linalg.gmres(G, I_vec, x0=I_vec, atol=1e-05, restart=15, maxiter=None, M=None, callback=my_callback, callback_type=None)
	et = time.time()
	elapsed = et - st
	print("Elapsed:", elapsed)
	print("Solution:", s)
	#res_graph = plt.plot(iterations, residuals, color = 'green')
	#plt.yscale('log')
	#plt.xlabel(r'$Iterations$', fontsize = 12) 
	#plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	
	S = s.reshape((N,N), order = 'F')
	return S

if __name__ == "__main__":
	GMRES_scipy(0.8, 4039)
	plt.show()
