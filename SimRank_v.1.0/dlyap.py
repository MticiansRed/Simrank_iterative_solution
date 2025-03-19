import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import sys
import time
from memory_profiler import profile

def SimpleIterIP(k_max, A, c, tau, N, eps = 1e-13):# OUTDATED/rework to call in GMRES_m similar manner.
	I = np.identity(N)
	A = csr_matrix(A)
	st = time.time()
	S = I
	residuals = []
	iterations = []
	for k in range(k_max):
		S_prev = S
		ATSA_factor = A.T@S@A
		S = (I-S + c*ATSA_factor - c*np.diag(np.diag(ATSA_factor)))*tau+S
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

#---main version---
def LSq(beta, H):
	m = H.shape[1]
	e1 = np.zeros((m+1,1))
	e1[0,0] = 1.0 #generating e1 vector
	b = e1*beta
	y = np.linalg.inv(H.T@H)@H.T@b
	return y

def Arnoldi(V_list, h_list, m_start, m, LinOp):
	for j in range(m_start, m):
		print(f"Building Arnoldi: V[:,{j}]", end = '\r')
		v_j = V_list[j]
		w_j = colvecto1dim(LinOp(v_j))
		for i in range(j):
			v_i = V_list[i]
			h_list[i][j] = v_i@w_j  
			w_j = w_j - h_list[i][j]*v_i
		w_j_norm2 = np.linalg.norm(w_j, ord = 2)
		h_list[j+1][j] = w_j_norm2
		if (w_j_norm2 == 0):
			return j
		V_list[j+1] = (w_j/w_j_norm2)
	return m

def colvecto1dim(u):
	return u.reshape(u.shape[0], order = 'F')

def GMRES_m(LinOp, m_Krylov, x_0, b, k_max, eps = 1e-13):
	N = x_0.shape[0] #N = n^2
	restarts = []
	residuals = []
	st = time.time()
	x = x_0
	for k in range(k_max):
		st_restart = time.time()
		r = b - LinOp(x)
		residual = np.linalg.norm(r, ord = 2)
		
		#Writing iteration information
		print("Residual:", residual) #residual with last restart solution.
		print("Relative residual:", residual/np.linalg.norm(LinOp(x), ord = 2))
		print("Restart:", k)
		restarts.append(k)
		residuals.append(residual)
		#
		
		if (residual < eps):
			break
		
		beta = np.linalg.norm(r, ord = 2)
		V_list = [np.zeros(N)] #Stores columns of V matrix
		V_list[0] = colvecto1dim(r)/beta
		H_list = [np.zeros(m_Krylov)] #Stores rows of Hessenberg matrix
		
		for m in range(1,m_Krylov):
			V_list.append(np.zeros(N)) #Reserving space for vector (column of V) v_{j+1}
			H_list.append(np.zeros(m_Krylov)) #Reserving space for row of H h_{j+1}
			st_iter = time.time()
			m = Arnoldi(V_list, H_list, (m-1), m,  LinOp)
			V = (np.array(V_list[:m])).T #Slicing V_list[:m] because v_{m+1} is not needed for projection step.
			H = (np.array(H_list)).T[:m].T #Slicing because everything right to m'th column is placeholding zeros.
			y = LSq(beta, H)
			x = x_0 + V@y
			
			# *break condition*
			et_iter = time.time()
			print(f"m = {m}; Residual = :", np.linalg.norm(b-LinOp(x), ord = 2), f"; Iteration time: {et_iter-st_iter} s")
		x_0 = x
		et_restart = time.time()
		print("Restart time:", et_iter - st_iter)

	et = time.time()
	elapsed = et - st
	
	#Writing solution information
	print(f"GMRES(m) time: {elapsed} s")
	print("Iterations", restarts)
	print("Residuals", residuals)
	#
	#plt.figure()
	#plt.grid()
	
	#Plotting
	res_graph = plt.plot(restarts, residuals, color = 'green')
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	#
	return x

def MinRes(k_max, A, c, N, eps = 1e-13): # OUTDATED/rework to call in GMRES_m similar manner.
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
