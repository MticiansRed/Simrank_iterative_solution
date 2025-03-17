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
	
def G(u, A, c, N): #Liear operator function. G(S) = I, G(S) = S-c*A.T@S@A+c*diag(A.T@S@A). u comes as !vector! and then converted to matrix.
	A = csr_matrix(A)
	U = u.reshape((N, N), order = 'F')
	ATUA = A.T@U@A
	G = U - c*ATUA+c*np.diag(np.diag(ATUA))
	G = G.reshape((N**2), order = 'F')
	return G

#---main version---
def Hessenberg(h, m):
	H = np.zeros((m+1, m))
	for i in range(m+1):
		for j in range(m):
			H[i,j] = h[i,j]
	return H
def LSq(beta, H):
	m = H.shape[1]
	e1 = np.zeros((m+1))
	e1[0] = 1.0 #generating e1 vector
	b = e1*beta
	y = np.linalg.inv(H.T@H)@H.T@b
	return y
def Arnoldi(V, W, h, m, A, c, N):
	for j in range(m):
		print(f"Building Arnoldi: V[:,{j}]", end = '\r')
		W[:,j] = G(V[:,j], A, c, N)
		for i in range(j):
			h[i,j] = V[:,i].T@W[:,j] #Note: .T to underline that is is a dot product
			W[:,j] = W[:,j] - h[i,j]*V[:,i]
		h[j+1,j] = np.linalg.norm(W[:,j],ord = 2)
		if (h[j+1,j] == 0):
			return j
		V[:,j+1] = W[:,j]/h[j+1,j]
	return m

def colvecto1dim(u):
	return u.reshape(u.shape[0], order = 'F')

def GMRES_m(k_max, m, A, c, N, eps = 1e-13):
	iterations = []
	residuals = []
	st = time.time()
	I_vec = np.identity(N).reshape((N**2), order = 'F')
	s = I_vec
	for k in range(k_max):
		st_iter = time.time()
		r = I_vec - G(s, A, c, N)
		
		
		iterations.append(k)
		print("Iteration:", k)
		residual = np.linalg.norm(I_vec - G(s, A, c, N), ord = 2)
		print("Residual:", residual)
		residuals.append(residual)
		if (residual < eps):
			break
		
		beta = np.linalg.norm(r, ord = 2)
		V = np.zeros((N**2,m+1)) #V = np.zeros((N**2,m))
		V[:,0] = r/beta #v_1 = r_0/beta
		W = np.zeros((N**2,m))
		h = np.zeros((m+1, m))
		m = Arnoldi(V, W, h, m, A, c, N)
		H = Hessenberg(h, m)
		y = LSq(beta, H)
		s = s + (V.T[:m].T)@y #Transposing because [:m] operation takes matrix by-rows.
		et_iter = time.time()
		print("Iteration time:", et_iter - st_iter)
	et = time.time()
	
	elapsed = et - st
	print(f"Elapsed: {elapsed} s")
	print("Iterations", iterations)
	print("Residuals", residuals)
	#plt.figure()
	#plt.grid()
	res_graph = plt.plot(iterations, residuals, color = 'green')
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	
	S = s.reshape((N,N), order = 'F')
	return S

def GMRES(k_max, m, A, c, N, eps = 1e-13): #classic
	
	iterations = []
	residuals = []
	st = time.time()
	
	I_vec = np.identity(N).reshape((N**2,1), order = 'F')
	s = I_vec
	r_0 = I_vec - G(s, A, c, N)
	
	iterations.append(0)
	print("Iteration:", 0)
	residual = np.linalg.norm(r_0, ord = 2)
	print("Residual:", residual)
	residuals.append(residual)
	
	r_0 = colvecto1dim(r_0) #reshaping to 1dim vector
	beta = np.linalg.norm(r_0, ord = 2)
	V = np.zeros((N**2,2)) #V = np.zeros((N**2,m))
	V[:,0] = r_0/beta #v_1 = r_0/beta
	W = np.zeros((N**2,1)) #originally create matrices of size according to maximum allowed Krylov dimensionality. 
	h = np.zeros((2, 1))
	for k in range(1,m):
		k = Arnoldi(V, W, h, k, A, c, N)
		H = Hessenberg(h,k)
		y = LSq(beta, H)
		s = I_vec + (V.T[:k].T)@y #s = s + .. ?
		r = I_vec - G(s, A, c, N)

		iterations.append(k)
		print("Iteration:", k)
		residual = np.linalg.norm(r, ord = 2)
		print("Residual:", residual)
		residuals.append(residual)
		if (residual < eps):
			break
		#expanding matrices
		V = np.concatenate((V, np.zeros((N**2, 1))), axis = 1)
		W = np.concatenate((W, np.zeros((N**2, 1))), axis = 1)
		print("h shape:", h.shape)
		h = np.concatenate((h, np.zeros((1, k))), axis = 0)
		print("h shape:", h.shape)
		h = np.concatenate((h, np.zeros((k+2, 1))), axis = 1)
		print("h shape:", h.shape)
	
	et = time.time()
	elapsed = et - st
	res_graph = plt.plot(iterations, residuals, color = 'green')
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	
	S = s.reshape((N,N), order = 'F')
	return S


def MinRes(k_max, A, c, N, eps = 1e-13): # G(S) = I, G(S) = S-c*A.T@S@A+c*diag(A.T@S@A)
	I = np.identity(N)
	I_vec = I.reshape((N**2, 1), order = 'F')
	st = time.time()
	s = I_vec
	residuals = []
	iterations = []
	
	r = I_vec - G(s, A, c, N)
	p = G(r, A, c, N)
	
	residual = np.linalg.norm(r, ord = 2)
	residuals.append(residual)
	iterations.append(0)
	print(f"Iteration: {0}")
	print(f"Residual: {residual}")
	
	for k in range(1,k_max):
		a = (r.T@r)/(p.T@p)
		s = s + a*r
		r = r - a*p
		p = G(r, A, c, N)
		iterations.append(k)
		#print(S)
		residual = np.linalg.norm(r, ord = 2)
		residuals.append(residual)
		print(f"Iteration: {k}")
		print(f"Residual: {residual}")
		if (residual < eps):
			break
	et = time.time()
	elapsed = et - st
	print(f"Elapsed: {elapsed} s")
	#plt.figure()
	#plt.grid()
	res_graph = plt.plot(iterations, residuals, color = 'red')
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	S = s.reshape((N,N), order = 'F')
	return S

def RNSD(k_max, A, c, N, eps = 1e-13): #Residual norm steepest descent
	I = np.identity(N)
	I_vec = I.reshape((N**2, 1), order = 'F')
	st = time.time()
	S = I_vec
	residuals = []
	iterations = []
	
	r = I_vec - G(S, A, c, N)
	
	residual = np.linalg.norm(r, ord = 2)
	residuals.append(residual)
	iterations.append(0)
	print(f"Iteration: {0}")
	print(f"Residual: {residual}")
	
	for k in range(1,k_max):
		v = G(r, A.T, c, N)
		Av = G(v, A, c, N)
		a = (v.T@v)/(Av.T@Av)
		S = S + a*v
		r = r - a*Av
		iterations.append(k)
		#print(S)
		residual = np.linalg.norm(r, ord = 2)
		residuals.append(residual)
		print(f"Iteration: {k}")
		print(f"Residual: {residual}")
		if (residual < eps):
			break
	et = time.time()
	elapsed = et - st
	print(f"Elapsed: {elapsed} s")
	#plt.figure()
	#plt.grid()
	res_graph = plt.plot(iterations, residuals, color = 'green')
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
