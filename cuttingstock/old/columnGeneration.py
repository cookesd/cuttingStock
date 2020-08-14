# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:24:01 2019

@author: cookesd
"""

from scipy.optimize import linprog
import scipy.optimize as opt
import numpy as np
import csv

##### knapsack solution #####
def knapsack(values, weights, capacity,solVect):
    '''Solves the unbounded knapsack problem using dynamic programming (recursion).
    The unbounded knapsack problem here tries to maximize the value (dual variable of the entering cut pattern)
    subject to the capacity constraints (the board cuts cannot exceed the board length).
    This new pattern will enter our basis if the value (dual var) is greater than 1,
    Otherwise, it will not improve the objective to the linear program.
    
    @param values (iterable of floats) the current dual variables for the linear programming solution (c_{B}B^{-1})
    @param weights (iterable of floats) the length of the desired cuts
    @param capacity (float) the knapsack capacity (length of the board)
    @param solVect (iterable of length number of cuts) can be a list of zeros initially; used for recursively calling knapsack
    '''
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
    
def my_callback(xk, **kwargs):
    """
    My "callback" function
    It prints and saves the tableau from the final solution
    (i.e., when kwargs['complete']=True)

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
    global objRow
    global dualVars
    global t
    global bvs
    global xs
    tableau = kwargs["tableau"]
    basis = kwargs["basis"]
    complete = kwargs["complete"]
    
    if complete:
        objRow = tableau[len(tableau)-1]
        dualVars = [abs(objRow[i]) for i in range(len(A[0]),len(A[0])+len(b))]
        t = tableau
        bvs = basis
        xs= xk
        
def printResult(resultDict):
    for pattern in resultDict:
        print('\n',resultDict[pattern]['patternQuantity'][0],' (',resultDict[pattern]['patternQuantity'][1],') Cuts of Pattern ',pattern[len(pattern)-1],':',sep='')
        for cut in resultDict[pattern]:
            if cut != 'patternQuantity' and cut != 'waste':
                print(resultDict[pattern][cut],'cuts of length',cut)
        print('with', resultDict[pattern]['waste'], 'units of waste')

##### main method #####
verbose = True
#infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\winstonExample(570).txt'
infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\consoleTable.txt'
lenDict = {}
boardLength = None
epsilon = .005
with open(infile) as f:
    boardLength = int(f.readline())
    for line in f.readlines():
        line=line.split(',')
        lenDict[line[0]]=int(line[1])
f.close()
if verbose:
    print('Board length:',boardLength)
    print('Length Dict:',lenDict)

A = np.array([[0]*len(lenDict.keys())]*len(lenDict.keys()))
b = np.array([0]*len(lenDict))
c = np.array([0]*len(lenDict))
sizes = np.array([0]*len(lenDict))
for (i,key) in zip(range(len(lenDict)),lenDict.keys()):
    A[i][i] = -boardLength/int(key) #makes a column (cut pattern var) in A matrix for each length in length dict
    b[i] = -lenDict[key] #the number of each length you need
    c[i] = 1 #cost for each board (cut pattern)
    #c[i] = boardLength + int(key)*A[i][i] #adding because A[i][i] is negative
    sizes[i] = int(key) #the size of the boards you need

t = None
objRow = None
dualVars = None
bvs = None
xs = None
res = linprog(c,A_ub = A,b_ub = b,callback=my_callback)
res
t
objRow
dualVars
bvs

benefit,knap = knapsack(dualVars,sizes,boardLength,[0]*len(sizes))
print('benefit:',benefit,'\nknapsack:',knap)
while benefit-1 > epsilon:
    newCol = np.array([[-item] for item in knap])
    A = np.append(A,newCol,axis=1)
    c = np.append(c,np.array([1])) #They're trying to min numBoards so all obj coeffs are 1
    res = linprog(c,A_ub = A,b_ub = b,callback=my_callback)
    print(t)
    benefit, knap = knapsack(dualVars,sizes,boardLength,[0]*len(sizes))
    print('benefit:',benefit,'\nknapsack:',knap)
#    end = input('Press enter to continue,q to quit')
#    if end == 'q':
#        benefit = -1
        
BT = np.array([[i[j] for i in A] for j in bvs]) # your B matrix for the final solution
B = BT.transpose()
xb = np.array([xs[j] for j in bvs])
cutDict = {}
for colNum in range(B.shape[1]):
    dictKey = 'pattern'+str(colNum+1)
    cutDict[dictKey]={'patternQuantity':(np.ceil(xb[colNum]),xb[colNum])}
    pattern = [(-B[j][colNum],sizes[j]) for j in range(len(sizes))] #(num cuts of that length, length of cut)
    waste = boardLength-sum([i*j for (i,j) in pattern])
    for cut in pattern:
        cutDict[dictKey][cut[1]]=cut[0]
    cutDict[dictKey]['waste']=waste
    
printResult(cutDict)

#def linprog_verbose_callback(xk, **kwargs):
#    """
#    A sample callback function demonstrating the linprog callback interface.
#    This callback produces detailed output to sys.stdout before each iteration
#    and after the final iteration of the simplex algorithm.
#
#    Parameters
#    ----------
#    xk : array_like
#        The current solution vector.
#    **kwargs : dict
#        A dictionary containing the following parameters:
#
#        tableau : array_like
#            The current tableau of the simplex algorithm.
#            Its structure is defined in _solve_simplex.
#        phase : int
#            The current Phase of the simplex algorithm (1 or 2)
#        nit : int
#            The current iteration number.
#        pivot : tuple(int, int)
#            The index of the tableau selected as the next pivot,
#            or nan if no pivot exists
#        basis : array(int)
#            A list of the current basic variables.
#            Each element contains the name of a basic variable and its value.
#        complete : bool
#            True if the simplex algorithm has completed
#            (and this is the final call to callback), otherwise False.
#    """
#    global c
#    tableau = kwargs["tableau"]
#    nit = kwargs["nit"]
#    pivrow, pivcol = kwargs["pivot"] #don't need
#    phase = kwargs["phase"] #don't need
#    basis = kwargs["basis"]
#    complete = kwargs["complete"]
#    
#    cb = []
#    for x in basis:
#        if x < len(c):
#            cb.append(c[x])
#        else:
#            cb.append(0)
#    B = [[tableau[i][x] for x in basis] for i in range(len(basis))]
#    #Binv = np.linalg.inv(B)
#    
#
#    saved_printoptions = np.get_printoptions()
#    np.set_printoptions(linewidth=500,
#                        formatter={'float': lambda x: "{0: 12.4f}".format(x)})
#    if complete:
#        print("--------- Iteration Complete - Phase {0:d} -------\n".format(phase))
#        print("Tableau:")
#    elif nit == 0:
#        print("--------- Initial Tableau - Phase {0:d} ----------\n".format(phase))
#
#    else:
#        print("--------- Iteration {0:d}  - Phase {1:d} --------\n".format(nit, phase))
#        print("Tableau:")
#
#    if nit >= 0:
#        print("" + str(tableau) + "\n")
#        if not complete:
#            print("Pivot Element: T[{0:.0f}, {1:.0f}]\n".format(pivrow, pivcol))
#        print("Basic Variables:", basis)
#        print()
#        print("Current Solution:")
#        print("x = ", xk)
#        print()
#        print("Current Objective Value:")
#        print("f = ", -tableau[-1, -1])
#        print()
#    np.set_printoptions(**saved_printoptions)




