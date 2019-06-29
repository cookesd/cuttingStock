# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:24:01 2019

@author: cookesd
"""

import pandas as pd
import numpy as np
import csv
from collections.abc import Iterable
import os

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
    
        
def findLV(Binv,b,a,tol = -1.0E-12):
    '''Finds the column of the leaving variable using the ratio test.
    (min_i{B^{-1}b_i/B^{-1}a_i})
    
    @param Binv, the inverse matrix of the current basis
    @param b, the original right hand side of the constraints
    @param a, the column vector of the entering variable
    returns the row of the leaving variable (also column in Binv) or identifies unboundedness'''
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
    print(lvrow)
    print('ratios',ratios)
    print('min ratio',minRatio)
    unbounded = minRatio < tol #the problem is unbounded if this minimum ratio is negative
    print('unbounded',unbounded)
    return(unbounded,lvrow[0][0],bbar,abar)

def updateBinv(Binv,abar,lvrow):
    matDim = len(Binv)
    eMat = np.identity(matDim) #identity matrix with same size as Binv
    newCol = -abar/abar[lvrow] #the lvrowth column should be -abar_ik/abar_rk with r,r element = 1/abar_rk
    newCol[lvrow] = 1/abar[lvrow]
    print(newCol)
    eMat[:,lvrow] = np.reshape(newCol,(1,matDim))
    
    newBinv = np.matmul(eMat,Binv)
    return(newBinv)
    
def calcDualVars(cB,Binv):
    cBBinv = np.matmul(cB,Binv)
    return(cBBinv)


#pre and post processing
def printResult(resultDict):
    for pattern in resultDict:
        print('\n',resultDict[pattern]['patternQuantity'][0],' (',resultDict[pattern]['patternQuantity'][1],') Cuts of Pattern ',pattern[len(pattern)-1],':',sep='')
        for cut in resultDict[pattern]:
            if cut != 'patternQuantity' and cut != 'waste':
                print(resultDict[pattern][cut],'cuts of length',cut)
        print('with', resultDict[pattern]['waste'], 'units of waste')

def buildModel(fName = None,bLength = None,lenDict = None):
    if fName == None:
        assert isinstance(bLength,int) or isinstance(bLength,Iterable), 'If no input file specified, you must supply the length of your cutting stock'
        if isinstance(bLength,int):
            bLength = list(bLength) #makes it a list (iterable) so can function same as multiple board lengths)
        assert isinstance(lenDict,dict) or isinstance(lenDict,pd.DataFrame), 'If no input file specified, you must supply the your desired cut sizes and quantities in a dict or pd.DataFrame'
        
    else:
        assert isinstance(fName,str), 'Filename must be a string'
        assert os.path.exists(fName), 'This is not a valid path'
    

    
        
        
##### main method #####
verbose = False
#infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\winstonExample(570).txt'
infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\consoleTable_n_plants.txt'
#infile = 'winstonExample(570).txt'
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
    print('B',B)
    print('bbar',bbar)
    print('cb',cB)
    print('lv',lv)
    print('benefit',benefit)
    input('press enter to continue')

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
    
printResult(cutDict)





