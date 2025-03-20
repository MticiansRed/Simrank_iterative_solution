import numpy as np
from scipy.sparse import csr_matrix
import sys
import time
import matplotlib.pyplot as plt 
import datetime as dt
from matplotlib.ticker import FixedLocator, FixedFormatter
from memory_profiler import profile
import dlyap

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

def Solve(acc, m_Krylov, k_iter_max, taskname, A, c, solvers): #solvers = list of flags: ['SimpleIter, GMRES, MinRes'] (in any order)
	n = A.shape[0]
	if (A.shape[0]!=A.shape[1]):
		print("Non-square matrix passed in argument. Stopped.")
		return 1
	I = np.identity(n) #identity matrix of required dimensions
	print("Adjacency matrix:")
	print(A)
	I = np.identity(n)
	I_vec = np.identity(n).reshape((n**2,1), order = 'F')
	A_csr = csr_matrix(A) #if A is already CSR -> changes nothing.
	G = G_operator(A_csr, c) #Initialize operator
	S = {} #init dict of solutions
	
	for solver in solvers:
		if (solver == "SimpleIter"):
			S_si = 0 #placeholder
			S["S_si"] = S_si
		elif (solver == "GMRES"):
			print(f"Starting GMRES with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			S_gmres = dlyap.GMRES_m(G, m_Krylov, I_vec, I_vec, k_iter_max, acc).reshape((n,n), order = 'F')
			S["S_gmres"] = S_gmres
		elif (solver == "GMRES_scipy"):
			print(f"Starting GMRES from SciPy with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			S_gmres_scipy = dlyap.GMRES_scipy(G, m_Krylov, I_vec, I_vec, k_iter_max, acc).reshape((n,n), order = 'F')
			S["S_gmres_scipy"] = S_gmres_scipy
		elif (solver == "MinRes"):
			print(f"Starting MinRes with {k_iter_max} iterations limit  ...")
			S_minres = 0#placeholder
			S["S_minres"] = S_minres

		else:
			print("Solver not found.")
			return 1
	for key in S:
		plt.savefig(taskname+key+"_eps_"+str(acc)+"_"+str(dt.datetime.now())+".png")
		plt.figure()
		graph = plt.imshow(np.log(S[key]-I+1e-15))
		cbar = plt.colorbar()
		cbar.set_label("ln(S[i,j])")
		plt.title(taskname)
		plt.title("Матрица S", fontweight = "bold")
	return S


