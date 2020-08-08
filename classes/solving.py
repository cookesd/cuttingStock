# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:24:01 2019

@author: cookesd
"""

from .CuttingStockSolution import CuttingStockSolution

def solve(self):
    """ Solves the cutting stock problem using the revised simplex method
    and dynamic programming to solve a knapsack problem for column generation
    
    @param infile (file path): file containing stock length, required cut sizes and quantities
    @param show (boolean): True if you want the verbose printout of the cutting stock solution
    @para graph (boolean): True if you want a bargraph showing the cut patterns
    
    returns a pd.DataFrame holding the final solution of cut patterns and their quantities
    """
    import pandas as pd
    import numpy as np
    from collections.abc import Iterable
    import os
    
    ##### knapsack solution #####
    def knapsack(values, weights, capacity,solVect = None):
        '''Solves the unbounded knapsack problem using dynamic programming (recursion).
        The unbounded knapsack problem here tries to maximize the value (dual variable of the entering cut pattern)
        subject to the capacity constraints (the board cuts cannot exceed the board length).
        This new pattern will enter our basis if the value (dual var) is greater than 1,
        Otherwise, it will not improve the objective to the linear program.
        
        @param values (iterable of floats) : knapsack obj function coefficitens (the current dual variables for the linear programming solution (c_{B}B^{-1}))
        @param weights (iterable of floats) : knapsack constraint coefficients for each cut (the length of the desired cuts)
        @param capacity (float) : the knapsack capacity (length of the board)
        @param solVect {optional} (iterable of length number of cuts) : should be a list of zeros initially;
        used for recursively calling knapsack; if no value specified, then automatically sets to list of zeros
        If a vector is applied, it is a starting knapsack solution
        
        returns finalSol : the solution to the knapsack ()
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
                solMat[i][i] +=1 #adding 1 cut of size "weight" to the solution matrix
                sol[i],solMat[i] = knapsack(values, weights, newCap,solMat[i]) #calls knapsack with the updated capacity after the new cut has been added
                
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
        
            
    def findLV(Binv,b,a,tol = -1.0E-12):
        '''Finds the column of the leaving variable using the ratio test.
        (min_i{B^{-1}b_i/B^{-1}a_i})
        
        @param Binv, the inverse matrix of the current basis
        @param b, the original right hand side of the constraints
        @param a, the column vector of the entering variable
        
        returns unbounded : boolean True if unboundedness detected; false o/w
        returns lvrow[0][0] : the row of the leaving variable from ratio test (lowest index chosen in case of ties)
        returns bbar : the costs of the basic variables
        returns abar : the column of the entering variable in the current basic feasible solution (B^{-1}*a_i for entering varable i)
        '''
        largeNum = max(b)**2 #this should be big enough, need to find better value
        bbar = np.matmul(Binv,b)
        abar = np.matmul(Binv,a)
        ratList = []
        for row in range(len(bbar)):
            if abar[row]<=0:
                ratList.append(largeNum)
            else:
                ratList.append(bbar[row][0]/abar[row][0])
            ratios = np.array(ratList) # gets elementwise quotient of the vectors
        lvrow = np.where(ratios==min(ratios)) #finds row of the minimum ratio (one that goes to zero fastest after pivot)
        minRatio = ratios[lvrow[0][0]] #the minimum ratio
    #    print('rows with min ratio',lvrow)
    #    print('ratios',ratios)
    #    print('min ratio',minRatio)
        unbounded = minRatio < tol #the problem is unbounded if this minimum ratio is negative
        return(unbounded,lvrow[0][0],bbar,abar)
    
    def updateBinv(Binv,abar,lvrow):
        '''
        Updates the B^{-1} matrix with the new col (abar) in place of the leaving column (same column as lvrow since square matrix)
        
        @param Binv : the B^{-1} matrix from the previous solution
        @param abar : the column for the entering basic variable when premultiplied by the current B^{-1} matrix
        (B^{-1}*a_i for new basic variable i)
        @para lvrow : the row corresponding to the leaving variable found using the ratio test (b_j / a_{ij}). In case of ties, lowest index chosen
        
        returns newBinv : the updated B^{-1} matrix
        '''
        matDim = len(Binv)
        eMat = np.identity(matDim) #identity matrix with same size as Binv
        newCol = -abar/abar[lvrow] #the lvrowth column (r) should be -abar_ik/abar_rk with (r,r) element = 1/abar_rk
        newCol[lvrow] = 1/abar[lvrow]
#        print('entering column\n', newCol)
        eMat[:,lvrow] = np.reshape(newCol,(1,matDim)) # places newCol into the lvrow column
        
        newBinv = np.matmul(eMat,Binv) #updates B^{-1} using E*B^{-1}
        return(newBinv)
        
    def calcDualVars(cB,Binv):
        '''Uses matrix multiplication to calculate c_B*B{^-1} (dual variables w for the current solution)
        
        @param cB : the cost coefficients of the basic variables
        @param Binv : the B^{-1} matrix for the current basis
        
        returns cBBinv : the current dual variables (w = c_B*B^{-1})
        '''
        cBBinv = np.matmul(cB,Binv)
        return(cBBinv)
    
    
    #pre and post processing
    
    def cleanResult(resultDF):
        '''Cleans the resulting DF for printing and plotting for user
        makes the useful pattern quantity an integer and rounds the actual quantity to 3 decimals
        makes each cut quantity an integer rounded to the nearest integer (fixes computer calculation rounding)
        
        @param resultDF (pd.DataFrame) : the result DataFrame from the cutting stock problem
        returns the cleaned pd.DataFrame'''
        
        clean = resultDF.copy()
        for p in clean.columns:
            for i in clean[p].index:
                if i == 'patternQuantity':
                    clean[p][i] = (np.array(int(clean[p][i][0])),np.round(clean[p][i][1],3))
                else:
                    clean[p][i] = int(np.round(clean[p][i],0))
        return(clean)
    
    
    
    def buildModel(fName = None,bLength = None,lenDict = None):
        if fName == None:
            assert isinstance(bLength,int) or isinstance(bLength,Iterable), 'If no input file specified, you must supply the length of your cutting stock'
            if isinstance(bLength,int):
                bLength = list(bLength) #makes it a list (iterable) so can function same as multiple board lengths)
            assert isinstance(lenDict,dict) or isinstance(lenDict,pd.DataFrame), 'If no input file specified, you must supply the your desired cut sizes and quantities in a dict or pd.DataFrame'
            
        else:
            assert isinstance(fName,str), 'Filename must be a string'
            assert os.path.exists(fName), 'This is not a valid path'
        
            
    ##### Function Main #####
    verbose = False
    
    lenDict = {key:value['reqQuant'] for key,value in self.cuts.items()}
    boardLength = self.boardLength
    epsilon = .005

    
    b = []
    Bdiag = []
    cutSizes = []
    Bdim = len(lenDict.keys())
    for key in lenDict.keys():
        Bdiag.append(np.floor(boardLength/int(key)))
        b.append([lenDict[key]])
        cutSizes.append(int(key))
    Bdiag = np.array(Bdiag)
    b = np.array(b)
    
    B = np.diag(Bdiag)
    cB = np.array([1]*Bdim)
     
    
    Binv = np.linalg.inv(B)
    dualVars = calcDualVars(cB,Binv)
    benefit,enteringCol = knapsack(dualVars,cutSizes,boardLength,[0]*len(cutSizes))
    enteringCol = np.reshape(enteringCol,(len(enteringCol),1)) #ensures this is column vector
    while benefit-1>epsilon:
        unbounded,lv,bbar,abar=findLV(Binv,b,enteringCol)
        if not unbounded:
            Binv = updateBinv(Binv,abar,lv)
            B = np.linalg.inv(Binv)
            dualVars = calcDualVars(cB,Binv)
            benefit,enteringCol = knapsack(dualVars,cutSizes,boardLength,[0]*len(cutSizes))
            enteringCol = np.reshape(enteringCol,(len(enteringCol),1)) #ensures this is column vector
        else:
            print('The problem is unbounded')
            benefit = 0
    
    unbounded,lv,bbar,abar=findLV(Binv,b,enteringCol)
    if not unbounded:
            Binv = updateBinv(Binv,abar,lv)
            B = np.linalg.inv(Binv)
            dualVars = calcDualVars(cB,Binv)
            benefit,enteringCol = knapsack(dualVars,cutSizes,boardLength,[0]*len(cutSizes))
    else:
        benefit=0
    
    cutDict = {}
    for colNum in range(len(bbar)):
        dictKey = 'Pattern'+str(colNum+1)
        cutDict[dictKey]={'patternQuantity':(np.ceil(bbar[colNum]),bbar[colNum])}
        pattern = [(B[j][colNum],cutSizes[j]) for j in range(len(cutSizes))] #(num cuts of that length, length of cut)
        waste = boardLength-sum([i*j for (i,j) in pattern])
        for cut in pattern:
            cutDict[dictKey][cut[1]]=cut[0]
        cutDict[dictKey]['waste']=waste
    
    cutDF = pd.DataFrame(cutDict)
    cleanDF = cleanResult(cutDF)
    res = CuttingStockSolution(cleanDF)
#    printResult(cleanDF)
#    plotResult(cleanDF)
    return(res)
#    return(cleanDF)

#next updates are to make this a module to call and make some tester files
#module needs to be able to:
    # receive board length and the required dict
    #return the final cleanDF
    #be able to call printResult and plotResult and pass cleanDF
    


