import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import sys
import time
from memory_profiler import profile


def GetRandomMatrix(N):
	A = np.random.rand(N,N) #demo matrix
	print("Random matrix:")
	print(A)
	return A

def G(u, A, c, N): #Liear operator function. G(S) = I, G(S) = S-c*A.T@S@A+c*diag(A.T@S@A). u comes as !vector! and then converted to matrix.
	A = csr_matrix(A)
	U = u.reshape((N, N), order = 'F')
	ATUA = A.T@U@A
	G = U - c*ATUA+c*np.diag(np.diag(ATUA))
	print("c*A.TUA-c*diag(A.TUA)")
	print(c*ATUA+c*np.diag(np.diag(ATUA)))
	G = G.reshape((N**2,1), order = 'F')
	return G
	

def main_procedure():
	n=3
	N = n**2
	A = GetRandomMatrix(N)*10
	A = A.T@A
	print(A)
	eigvals, eigvecs = np.linalg.eig(A.T)
	print("eigvecs cols:")
	print(eigvecs) 
	
	u = (eigvecs[0].reshape((N,1), order = 'F')).T
	print("u", u)
	v = (eigvecs[1].reshape((N,1), order = 'F')).T
	print("v", v)
	S = u.T@v
	print(S)
	print(S)
	#tmp = G((u.T@v).reshape((N**2,1), order = 'F'), A, 0.8, N)
	

if __name__=="__main__":
	main_procedure()

