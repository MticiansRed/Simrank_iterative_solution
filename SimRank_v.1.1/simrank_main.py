import sys
import fb
import dlyap
import launch_solver as slv
import matplotlib.pyplot as plt 

def process_argv(argv, d):
	argc = len(argv)
	if (argc == 1):
		print("You have not used any flags. Use 'default' flag to load values from program;\n"\
		"Write in format:\n python3 *program*.py acc *value* m *value* k_max *value* taskname *task name* c *value* solvers *solver 1* *solver 2* ..."\
		"\nAvailable solvers: SimpleIter ; GMRES ; GMRES_scipy ; MinRes")
		return 1
	if (argc>1 and argv[1] == "default"):
		return 0
	if (argc>2):
		for i_arg in range(argc):
			if (argv[i_arg] == "acc"):
				d["acc"] = float(argv[i_arg+1])
			if (argv[i_arg] == "m_Krylov"):
				d["m"] = int(argv[i_arg+1])
			if (argv[i_arg] == "k_max"):
				d["k_max"] = int(argv[i_arg+1])
			if (argv[i_arg] == "taskname"):
				d["taskname"] = argv[i_arg+1]
			if (argv[i_arg] == "c"):
				d["c"] = float(argv[i_arg+1])
			if (argv[i_arg] == "solvers"):
				solvers_list = []
				for i in range(i_arg+1, argc):
					solvers_list.append(argv[i])
				d["solvers"] = solvers_list
		print(d["solvers"])
		return 0
	return 1

def main_process(args):
	d = {"acc":args[0], "m_Krylov": args[1], "k_max": args[2], "taskname": args[3], "c": args[4], "solvers":args[5]}
	if (process_argv(sys.argv, d)):
		return
	print("Arguments:\n", d)
	acc = d["acc"]
	m_Krylov = d["m_Krylov"]
	k_iter_max = d["k_max"]
	taskname = d["taskname"]
	c = d["c"]
	solvers = d["solvers"]
	plt.figure()
	plt.grid()
	if (taskname == "Fb"):
		A = fb.ObtainMatrix()
	slv.Solve(acc, m_Krylov, k_iter_max, taskname, A, c, solvers)
	plt.show()

if __name__=="__main__":
	args = [1e-5,15,1000000000, "Fb", 0.8,  ["GMRES_scipy"]] #default args
	main_process(args)
