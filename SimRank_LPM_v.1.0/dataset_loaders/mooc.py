import pandas as pd
import numpy as np
import scipy.sparse as scsp
import matplotlib.pyplot as plt

def GetMatrix(path, showfig, delimeter = '\t'):
	df = pd.read_csv(path, sep=delimeter)
	usridlist = np.array( df["USERID"] )
	tgtidlist = np.array( df["TARGETID"] )
	
	row_inds = []
	col_inds = []
	data = []
	
	users_number = np.max(usridlist)
	tgts_number = np.max(tgtidlist)
	print(f"Largest USR index : {users_number}")
	print(f"Largest TGT index : {tgts_number}")
	tgtidlist += users_number + 1 # shift indices 
	
	for usrid, tgtid in zip(usridlist, tgtidlist) :
		i = usrid
		j = tgtid
		row_inds.append(i)
		col_inds.append(j)
		data.append(1.)
	
	n = users_number + tgts_number + 2
	print(f"Expected matrix dimensionality : {n} x {n}")
	print(np.max(row_inds))
	print(np.max(col_inds))
	A = scsp.coo_matrix( (data, (row_inds, col_inds)), shape=(n,n))
	#print(f"A - A.T : {np.linalg.norm(A.toarray() - A.T.toarray())}")
	if showfig:
		plt.figure()
		plt.imshow(A.toarray(), cmap = 'binary')
		plt.show()
	return A.tocsr()

def norm1_ColumnNormalize(M): #may be optimized! L1 col norms can be easily obtained by sum(A).
	col_1_norms = np.sum(np.abs(M), axis = 0)
	col_1_norms[col_1_norms == 0] = 1 #Avoid div by 0
	print("Columns 1-norms:")
	print(col_1_norms)
	normalized = M/col_1_norms
	print("Column 1-normalized matrix:")
	print (normalized)
	return normalized

def ObtainMatrix(path = "data/mooc_actions.tsv", showfig=0):
	A = norm1_ColumnNormalize(GetMatrix(path=path, showfig=showfig))
	return A
