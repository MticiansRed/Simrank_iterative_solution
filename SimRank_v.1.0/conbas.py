import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags

def G(u, A, c, N): #Liear operator function. G(S) = I, G(S) = S-c*A.T@S@A+c*diag(A.T@S@A). u comes as !vector! and then converted to matrix.
	A = csr_matrix(A)
	U = u.reshape((N, N), order = 'F')
	ATUA = A.T@U@A
	G = U - c*ATUA+c*np.diag(np.diag(ATUA))
	G = G.reshape((N**2,1), order = 'F')
	return G

def colvecto1dim(u):
	return u.reshape(u.shape[0], order = 'F')

def norm1_ColumnNormalize(M): #may be optimized! L1 col norms can be easily obtained by sum(A).
	col_1_norms = np.sum(np.abs(M), axis = 0)
	col_1_norms[col_1_norms == 0] = 1 #Avoid div by 0
	print("Columns 1-norms:")
	print(col_1_norms)
	normalized = M/col_1_norms
	print("Column 1-normalized matrix:")
	print (normalized)
	return normalized

def Sparsity(A):
	m, n = A.shape[0], A.shape[1]
	n_zeros = 0
	for i in range(m):
		for j in range(n):
			if (A[i,j] == 0):
				n_zeros = n_zeros + 1
	return n_zeros/(m*n)

def diagG(u,v):
	d = u.T.multiply(v)
	return spdiags(d.toarray(), 0, format = "csr")

def GEfest(j, A, c, N): #G, E, effective, fast - GEfest! O(N^2) way to compute G(e_j) instead of O(2*N^3). 
	#st = time.time() #somewhere lil needs to be used.
	i = ( N//((N+1)-(j+1)%N) )-1
	j_hat = ( (j+1)%N-1)
	A = csr_matrix(A)
	#et = time.time()
	#print("Elapsed 1:", et-st)
	#st = time.time()
	#E_j = e_j.reshape((N, N), order = 'F')
	E_j = csr_matrix((N,N))
	E_j[j_hat,i]
	#et = time.time()
	#print("Elapsed 2:", et-st)
	#st = time.time()
	u = A[j_hat,:].T #transpose is needed due to the fact that scipy slices produce 2-dim vectors.
	#et = time.time()
	#print("Elapsed 3:", et-st)
	#st = time.time()
	v = A[i,:]
	#et = time.time()
	#print("Elapsed 4:", et-st)
	#st = time.time()
	ATE_jA = u@v
	#et = time.time()
	#print("Elapsed 5:", et-st)
	#st = time.time()
	#G = E_j - c*ATE_jA+c*csr_diag(ATE_jA)
	G = E_j - c*ATE_jA + c*diagG(u,v)
	#et = time.time()
	#print("Elapsed 6:", et-st)
	G = G.reshape((N**2,1), order = 'F')
	return G

def ConstructBasisGEfest(A, c, N, offset=0):
	G_matrix = csc_matrix((N**2,N**2))
	print("Constructing G(s) basis...")
	rank = N**2-offset
	for j in range(rank):
		print("Constructing G[:,j], j = ", j+1, " of ", rank, end = '\r')
		#e_j = csr_matrix((N**2,1))
		#e_j[j,0] = 1
		#st = time.time()
		G_matrix[:,j] = GEfest(j, A, c, N)
		#et = time.time()
		#print("Basis j:", j, " of ", rank, " elapsed: ", et-st)
	print("Basis constructed.")
	#G_matrix.dump("G_basis_Fb.dat")
	return G_matrix.tocsr() #returns csr!

def ConstructBasis(A, c, N):
	I_vec = np.identity(N).reshape((N**2,1), order = 'F')
	#G_matrix = np.zeros((N**2,N**2))
	G_matrix = lil_matrix((N**2,N**2))
	print("Constructing G(s) basis...")
	for j in range(N**2):
		print("Constructing G[:,j], j = ", j, " of ", N**2, end = '\r')
		e_j = np.zeros((N**2,1))
		e_j[j,0] = 1
		G_matrix[:,j] = colvecto1dim(G(e_j, A, c, N))
	print("Basis constructed. Basis matrix sparsity: ", Sparsity(G_matrix))
	return G_matrix.tocsr() #returns csr!

def crashtest(A, c, N, G_basis, tries):
	max_err_abs = 0.0
	max_err_rel = 0.0
	for i in range(tries):
		print("Test: ", i, end = '\r')
		if (i%1000 == 0):
			print("Test: ", i)
		u = np.random.randint(low = 0, high = 1001, size = (N**2, 1))/1000
		s_full = G(u, A, c, N)
		s_approx = G_basis@u
		err_abs = np.linalg.norm((s_full-s_approx), ord = 'fro')
		err_rel = err_abs/np.linalg.norm(s_full, ord = 'fro')
		if (max_err_abs < err_abs):
			max_err_abs = err_abs
		if (max_err_rel< err_rel):
			max_err_rel = err_rel
		#print("s_1-s_2 norm = ", err)
	print("Maximum error abs:", max_err_abs)
	print("Maximum error rel:", max_err_rel)
	
if __name__=="__main__":
	N = 3
	c = 0.8
	for tests in range(10):
		A = np.random.randint(low=0, high=2, size=(N,N))
		#A = norm1_ColumnNormalize(A)
		print("Matrix A:")
		print(A)
		print(A.shape)
		print("A sparsity: ", Sparsity(A))
		G_basis = ConstructBasisGEfest(A, c, N)
		print(G_basis.todense())
		G_basis = ConstructBasis(A, c, N)
		print(G_basis.todense())
		'''
		for d in range(10):
			G_basis = ConstructBasisGEfest(A, c, N)
			print("G basis shape::")
			print(G_basis.shape)
			print("G sparsity: ", Sparsity(G_basis))
			#eigvals = np.linalg.eigvals(G_basis)
			#print("G_basis eigvals:", eigvals)
			crashtest(A, c, N, G_basis, 10**4)
		'''
	
