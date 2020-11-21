# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:54:52 2019

Tests the methods of the CuttingStock class
@author: dakar
"""

import classes as cs # runs the init in classes dir
# from classes import CuttingStock as cs
# import os

#%% Build sample data and cutting stock problem
# Cutting Stock tester

cutDict = {0: {'reqQuant': 5, 'attr2': 1, 'attr3': 'sdfs'},
           1: {'reqQuant': 10, 'attr2': 47}}
cs1 = cs.CuttingStock(cutDict,10)

### testing CuttingStock Class methods ###
updateCutDict = {1:{'attr2':13,'attr3':12},0:{'attr3':50}}
addCutDict = {2:{'reqQuant':15}}
delCutDict = 2
bl = 240

#%% Build additional problems and test change functions
cs2 = cs1.copy()
#prints original cut dict and boardlength
print('Original cuts: \n{}'.format(cs1.cuts))
print('Original board length: {}'.format(cs1.boardLength))

#changes cut dict and prints it
cs1.update_cuts(updateCutDict)
print('Updated cuts: \n{}'.format(cs1.cuts))

# adds a cut and prints new cut dict
cs1.add_cuts(addCutDict)
print('With new cut: \n{}'.format(cs1.cuts))

# removes a cut and prints new cut dict
cs1.remove_cuts(delCutDict)
print('With cut removed: \n{}'.format(cs1.cuts))

# sets the boardLength to 240 and prints it
cs1.set_boardLength(bl)
print('No change to cuts: \n{}'.format(cs1.cuts))
print('New board length: {}'.format(cs1.boardLength))

# prints the copy to make sure it has the original values
print('Copy should contain original cuts: \n{}'.format(cs2.cuts))
print('Copy should contain original board length: {}'.format(cs2.boardLength))



#%% Tests the Cutting Stock algorithm
test_length_dict = {}
#must run the whole file for to get the right directory
#or change the cwd to the desired directory
with open('./data/consoleTable.txt') as f:
# with open(os.path.join(os.path.dirname(os.path.realpath('cs_tester.py')),'./data/consoleTable.txt')) as f:
    test_board_length = int(f.readline())
    for line in f.readlines():
        line=line.split(',')
        test_length_dict[line[0]]=int(line[1])
    f.close()

testCS = cs.CuttingStock(test_length_dict,test_board_length)
print(testCS)
res = testCS.solve()

#%% Tests the CuttingStockSolution class
print(res.cut_plot) # shows the plot of cut patterns
print(res) # print the result info
df = res.to_DataFrame()
print(df)
print(res.sol)
