# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:24:01 2019

@author: cookesd
"""

from scipy.optimize import linprog
import scipy.optimize as opt
import numpy as np

c = [1.5,2.15]
A = [[1,1]]
b = [210]
x0bounds = (70,90)
x1bounds = (100,140)

res = linprog(c,A,b,bounds=(x0bounds,x1bounds),method='simplex',callback=opt.linprog_verbose_callback)


def linprog_verbose_callback(xk, **kwargs):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces detailed output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    xk : array_like
        The current solution vector.
    **kwargs : dict
        A dictionary containing the following parameters:

        tableau : array_like
            The current tableau of the simplex algorithm.
            Its structure is defined in _solve_simplex.
        phase : int
            The current Phase of the simplex algorithm (1 or 2)
        nit : int
            The current iteration number.
        pivot : tuple(int, int)
            The index of the tableau selected as the next pivot,
            or nan if no pivot exists
        basis : array(int)
            A list of the current basic variables.
            Each element contains the name of a basic variable and its value.
        complete : bool
            True if the simplex algorithm has completed
            (and this is the final call to callback), otherwise False.
    """
    global c
    tableau = kwargs["tableau"]
    nit = kwargs["nit"]
    pivrow, pivcol = kwargs["pivot"] #don't need
    phase = kwargs["phase"] #don't need
    basis = kwargs["basis"]
    complete = kwargs["complete"]
    
    cb = []
    for x in basis:
        if x < len(c):
            cb.append(c[x])
        else:
            cb.append(0)
    B = [[tableau[i][x] for x in basis] for i in range(len(basis))]
    Binv = np.linalg.inv(B)
    

    saved_printoptions = np.get_printoptions()
    np.set_printoptions(linewidth=500,
                        formatter={'float': lambda x: "{0: 12.4f}".format(x)})
    if complete:
        print("--------- Iteration Complete - Phase {0:d} -------\n".format(phase))
        print("Tableau:")
    elif nit == 0:
        print("--------- Initial Tableau - Phase {0:d} ----------\n".format(phase))

    else:
        print("--------- Iteration {0:d}  - Phase {1:d} --------\n".format(nit, phase))
        print("Tableau:")

    if nit >= 0:
        print("" + str(tableau) + "\n")
        if not complete:
            print("Pivot Element: T[{0:.0f}, {1:.0f}]\n".format(pivrow, pivcol))
        print("Basic Variables:", basis)
        print()
        print("Current Solution:")
        print("x = ", xk)
        print()
        print("Current Objective Value:")
        print("f = ", -tableau[-1, -1])
        print()
    np.set_printoptions(**saved_printoptions)



##### knapsack solution #####
def knapsack(values, weights, capacity,solVect):
#    if verbose:
#        print(solVect)
    solMat = np.array([solVect]*len(values))
    sol = [0]*len(values) #solution to the subproblem (capacity-values[i]) after adding item i to knapsack
    largerSol = [0]*len(values) #solution to subproblem plus adding item i
    finalSol = None
    # finds the max value for the subproblem with capacity (after removing capacity for that item)
    for i in range(len(values)):
        if weights[i] <= capacity:
            newCap = capacity-weights[i]
            solMat[i][i] +=1 #####this part isn't working
            sol[i],solMat[i] = knapsack(values, weights, newCap,solMat[i])
            
        else:
            sol[i]=0
    # finds the solution to the current problem
    for i in range(len(values)):
        if weights[i] <= capacity:
            largerSol[i] = sol[i] + values[i]
        else:
            largerSol[i] = 0
    addedItem = largerSol.index(max(largerSol)) #finds the item to add into knapsack(item with largest value)
    finalSol = largerSol[addedItem]
    return(finalSol,solMat[addedItem])


vals = [11,7,12]
weight = [4,3,5]
cap = 10
#vals=[4,1]
#weight = [3,2]
#cap = 3
x = [0]*len(vals)
benefit,knap = knapsack(vals,weight,cap,x)
benefit
knap
