import numpy as np
import simrank_main as sm
import datetime as dt
import argparse

dateformat = "%Y-%m-%d-%H-%M"
path = "results/solutions"


def getsolution(taskname, c, solver, maxiter):
	args = {
	"acc":1e-15,
	"m_Krylov": 15, 
	"rank": 200, 
	"k_iter_max": maxiter, 
	"taskname": taskname, 
	"c": c, 
	"solver": solver,  
	"optimize" : True}
	#args = sm.load_args(sm.proc_args().argsfrom)
	S = sm.launch(args)
	np.save(
	f"{path}/S_{args['taskname']}_c_{args['c']}_solver_{solver}_{dt.datetime.now().strftime(dateformat)}.npy", S)

if __name__ == "__main__":
	#Note: ranks:
	#ranks_metro = np.arange(10, 304, 10)
	#ranks_eumail = np.arange(100, 1006, 100)
	#ranks_fb = np.arange(100, 4097, 100)
	parser = argparse.ArgumentParser("collect_iterative_solutions.py")
	parser.add_argument("-tn", "--taskname", help="specify task name, for options see simrank_main")
	parser.add_argument("-sl", "--solver", help="specify solver, for options see simrank_main")
	parser.add_argument("-c", "--cparam", help="Usage: c parameter value")
	parser.add_argument("-m", "--mKrylov", help="Krylov solver max iterations in restart")
	parser.add_argument("-it", "--maxiter", help = "Specify maxiter")
	clargs = parser.parse_args()
	solver = clargs.solver
	taskname = clargs.taskname
	c = float(clargs.cparam)
	maxiter = int(clargs.maxiter)
	getsolution(taskname, c, solver, maxiter)

