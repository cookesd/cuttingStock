# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 10:23:48 2019

@author: cookesd
"""

import os
os.chdir(r'C:\Users\dakar\Desktop\cookesdRepos\cuttingstock\cuttingstock')


#infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\winstonExample(570).txt'
#infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\consoleTable_n_plants.txt'
#infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\dad_simple_test.txt'
infile = r'C:\Users\dakar\Desktop\cookesdRepos\cuttingstock\cuttingstock\consoleTable.txt'
#infile = 'winstonExample(570).txt'

from cutting_stock import cuttingStockProblem as cs

res = cs(infile)

###need to first make an initial cutting stock problem object
###then using the object need to run the problem getting the result object
###then using the object be able to call the plotting and printing methods