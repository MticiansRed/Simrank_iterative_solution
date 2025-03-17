import fb
import metro
import dlyap_fast
import artificial 
import matplotlib.pyplot as plt 

if __name__=="__main__":
	
	plt.figure()
	plt.grid()
	fb.Facebook(1e-5, 15)
	#artificial.SolveDLyap(1e-15, 50, 1.48, 500) #1.48 is experimentally optimal.
	#fb.showmatrix()
	plt.show()
	
