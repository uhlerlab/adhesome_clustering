# Import libraries
import numpy as np
from scipy import sparse
import scipy.stats as ss
import pandas as pd
import networkx as nx
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import math
import time
from tqdm import tqdm
import random
from ortools.linear_solver import pywraplp


def main():
    
    # Get adjacency matrix and form w_plus, w_minus
    print('Load graph and form edge costs')
    G = pickle.load(open('graph_0.pkl', 'rb'))
    A = nx.adjacency_matrix(G, weight='scaled_hic').todense()
    np.fill_diagonal(A,1)
    w_plus = np.array(A)
    w_minus = 1-w_plus
    w_pm = w_plus-w_minus
    n = G.number_of_nodes()
    
    # Create the MIP solver with the SCIP backend.
    print('Create MIP solver')
    solver = pywraplp.Solver.CreateSolver('SCIP')
    infinity = solver.infinity()

    # Define variables
    print('Define variables')
    n = len(G.nodes)
    x = []
    for i in np.arange(0,n,1):
        for j in np.arange(i+1,n,1):
            x.append(solver.IntVar(0.0, 1.0, 'x_'+str(i)+'_'+str(j)))
    print('Number of variables =', solver.NumVariables())

    # Define constraints
    print('Define constraints')
    for i in np.arange(0,n,1):
        for j in np.arange(i+1,n,1):
            for k in np.arange(j+1,n,1):
                ij = int((n*(n-1)-(n-i)*(n-i-1))/2+j-i)-1
                jk = int((n*(n-1)-(n-j)*(n-j-1))/2+k-j)-1
                ik = int((n*(n-1)-(n-i)*(n-i-1))/2+k-i)-1
                solver.Add( (1-x[ij]) + (1-x[jk]) >= (1-x[ik]))
                solver.Add( (1-x[ij]) + (1-x[ik]) >= (1-x[jk]))
                solver.Add( (1-x[ik]) + (1-x[jk]) >= (1-x[ij]))
    print('Number of constraints =', solver.NumConstraints())

    # Define objective
    print('Define objective')
    objective = 0
    for i in np.arange(0,n,1):
        for j in np.arange(i+1,n,1):
            ij = int((n*(n-1)-(n-i)*(n-i-1))/2+j-i)-1
            objective += x[ij]*w_minus[i,j] + (1-x[ij])*w_plus[i,j]

    # Minimize objective
    print('Minimize objective')
    solver.Minimize(objective)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
    else:
        print('The problem does not have an optimal solution.')
    print('Advanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

    # Create and save adjacency matrix of resulting clique graph
    print('Create and save clique graph adjacency matrix')
    solution_adj_mat = np.zeros((n,n))
    for i in np.arange(0,n,1):
        for j in np.arange(i+1,n,1):
            solution_adj_mat[i,j] = x[int((n*(n-1)-(n-i)*(n-i-1))/2+j-i)-1].solution_value()
            solution_adj_mat[j,i]= solution_adj_mat[i,j]
    pickle.dump(solution_adj_mat, open('solution_adj_mat.pkl', 'wb'))


if __name__ == "__main__":
    main()