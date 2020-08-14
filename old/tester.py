# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 10:23:48 2019

@author: cookesd
"""

#infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\winstonExample(570).txt'
#infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\consoleTable_n_plants.txt'
#infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\dad_simple_test.txt'
infile = r'C:\Users\cookesd\Desktop\cookesdRepos\cuttingStock\dad_dims_ints.txt'
#infile = 'winstonExample(570).txt'

from cuttingStock import cuttingStockProblem as cs

cs(infile)

###need to first make an initial cutting stock problem object
###then using the object need to run the problem getting the result object
###then using the object be able to call the plotting and printing methods
