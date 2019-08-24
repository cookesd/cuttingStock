# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:54:52 2019

Tests the methods of the CuttingStock class
@author: dakar
"""

from my_packages import CuttingStock as cs

# Cutting Stock tester

cutDict = {0: {'reqQuant': 5, 'attr2': 1, 'attr3': 'sdfs'},
           1: {'reqQuant': 10, 'attr2': 47}}
cs1 = cs.CuttingStock(cutDict,10)

### testing CuttingStock Class methods ###
updateCutDict = {1:{'attr2':13,'attr3':12},0:{'attr3':50}}
addCutDict = {2:{'reqQuant':15}}
delCutDict = 2
bl = 240


cs2 = cs1.copy()
#prints original cut dict and boardlength
print(cs1.cuts)
print(cs1.boardLength)

#changes cut dict and prints it
cs1.update_cuts(updateCutDict)
print(cs1.cuts)

# adds a cut and prints new cut dict
cs1.add_cuts(addCutDict)
print(cs1.cuts)

# removes a cut and prints new cut dict
cs1.remove_cuts(delCutDict)
print(cs1.cuts)

# sets the boardLength to 240 and prints it
cs1.set_boardLength(bl)
print(cs1.cuts)
print(cs1.boardLength)

# prints the copy to make sure it has the original values
print(cs2.cuts)
print(cs2.boardLength)


#tests the Cutting Stock algorithm
ld = {}
fName = r'C:\Users\dakar\Desktop\cookesdRepos\cuttingstock\cuttingstock\consoleTable_n_plants.txt'
with open(fName) as f:
    bl = int(f.readline())
    for line in f.readlines():
        line=line.split(',')
        ld[line[0]]=int(line[1])
    f.close()

testCS = cs.CuttingStock(ld,bl)
print(testCS)
res = cs.cuttingStockProblem(testCS)

# tests the CuttingStockSolution class
res.plotResult()
res.printResult()
df = res.to_DataFrame()
print(df)
print(res.sol)
