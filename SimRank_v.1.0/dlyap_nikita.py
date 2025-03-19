import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import sys
import time
from memory_profiler import profile


def SimpleIter(k_max, A, c, N, eps = 1e-13): #in most models c = 0.8 and k_max = 5
	I = np.identity(N)
	A = csr_matrix(A)
	st = time.time()
	S = I
	residuals = []
	iterations = []
	for k in range(k_max):
		S_prev = S
		ATSA_factor = A.T@S@A
		S = c*(ATSA_factor)-c*np.diag(np.diag(ATSA_factor))+I
		residual = np.linalg.norm(S-S_prev, ord = 'fro') #fro? l2?
		residuals.append(residual)
		iterations.append(k)
		print(f"Iteration: {k}")
		print(f"Residual: {residual}")
		#print(S)
		if (residual < eps):
			break
	et = time.time()
	elapsed = et - st
	print(f"Elapsed: {elapsed} s")
	#plt.figure()
	#plt.grid()
	res_graph = plt.plot(iterations, residuals)
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	
	return S
	
def SimpleIterIP(k_max, A, c, ip, N, eps = 1e-13): #in most models c = 0.8 and k_max = 5
	I = np.identity(N)
	A = csr_matrix(A)
	st = time.time()
	S = I
	residuals = []
	iterations = []
	for k in range(k_max):
		S_prev = S
		ATSA_factor = A.T@S@A
		S = ( c*(ATSA_factor)-c*np.diag(np.diag(ATSA_factor))+I )*ip
		residual = np.linalg.norm(S-S_prev, ord = 'fro') #fro? l2?
		residuals.append(residual)
		iterations.append(k)
		print(f"Iteration: {k}")
		print(f"Residual: {residual}")
		#print(S)
		if (residual < eps):
			break
	et = time.time()
	elapsed = et - st
	print(f"Elapsed: {elapsed} s")
	#plt.figure()
	#plt.grid()
	res_graph = plt.plot(iterations, residuals)
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	
	return S
	

def G(u, A): #Liear operator function. G(S) = I, G(S) = S-c*A.T@S@A+c*diag(A.T@S@A). u comes as !vector! and then converted to matrix.
	#A = csr_matrix(A)
	return A@u

#---main version---
def Hessenberg(h, m):
	H = np.zeros((m+1, m))
	for i in range(m+1):
		for j in range(m):
			H[i,j] = h[i,j]
	return H
def LSq(beta, H):
	m = H.shape[1]
	e1 = np.zeros((m+1,1))
	e1[0,0] = 1.0 #generating e1 vector
	b = e1*beta
	y = np.linalg.inv(H.T@H)@H.T@b
	return y
def Arnoldi(V, W, h, m, A):
	for j in range(m):
		print(f"Building Arnoldi: V[:,{j}]", end = '\r')
		W[:,j] = colvecto1dim(G(V[:,j], A))
		for i in range(j):
			h[i,j] = V[:,i]@W[:,j] 
			W[:,j] = W[:,j] - h[i,j]*V[:,i]
		h[j+1,j] = np.linalg.norm(W[:,j],ord = 2)
		if (h[j+1,j] == 0):
			return j
		V[:,j+1] = W[:,j]/h[j+1,j]
	return m

def colvecto1dim(u):
	return u.reshape(u.shape[0], order = 'F')

def GMRES_m(k_max, m, A, N, eps = 1e-13):
	iterations = []
	residuals = []
	st = time.time()
	b = np.zeros((N,1))
	s = np.array([[0.002, 1000000.0, 100000000.0, 1e-06]]).T
	for k in range(k_max):
		st_iter = time.time()
		r = b - G(s, A)
		
		
		iterations.append(k)
		print("Iteration:", k)
		residual = np.linalg.norm(b - G(s, A), ord = 2)
		print("Residual:", residual)
		print("Relative residual:", residual/np.linalg.norm(G(s, A), ord = 2))
		residuals.append(residual)
		if (residual < eps):
			break
		
		beta = np.linalg.norm(r, ord = 2)
		V = np.zeros((N,m+1)) #V = np.zeros((N**2,m))
		V[:,0] = colvecto1dim(r)/beta #v_1 = r_0/beta
		W = np.zeros((N,m))
		h = np.zeros((m+1, m))
		m = Arnoldi(V, W, h, m, A)
		H = Hessenberg(h, m)
		y = LSq(beta, H)
		s = s + (V.T[:m].T)@y #Transposing because [:m] operation takes matrix by-rows.
		et_iter = time.time()
		print("Iteration time:", et_iter - st_iter)
	et = time.time()
	
	elapsed = et - st
	print(f"Elapsed: {elapsed} s")
	#print("Iterations", iterations)
	#print("Residuals", residuals)
	#plt.figure()
	#plt.grid()
	res_graph = plt.plot(iterations, residuals, color = 'green')
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	
	return s
