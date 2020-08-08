# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:36:31 2020

@author: dakar
"""

import pyomo.environ as pyo
import pandas as pd

infile = r'C:\Users\dakar\Desktop\cookesdRepos\cuttingStock\cuttingstock\consoleTable.txt'

board_length = None
cut_sizes = {}
with open(infile) as f:
    board_length = int(f.readline())
    for i,line in enumerate(f.readlines()):
        line_list = line.split(',')
        cut_sizes[i] = {'length':int(line_list[0]),
                        'quant':int(line_list[1])}

cut_sizes = pd.DataFrame(cut_sizes).transpose()

class CuttingStockOpt(object):
    def __init__(self,in_file = None,board_length = None, cut_sizes = {},init_patterns=pd.DataFrame()):
        
        self.board_length = board_length
        self.cut_sizes = cut_sizes
        self.pattern_size_quant_df = init_patterns
        self.in_file = in_file
        if self.in_file:
            self.read_data(in_file) # initializes self.board_length and self.cut_sizes
            self.build_init_patterns()

    def read_data(self,in_file):
        '''
        Reads data from the supplied input file
        First row is the board/stock length
        Remaining rows are comma-separated cut-length,desired quantity tuples.
        

        Parameters
        ----------
        in_file : File Path
            Path to input file containing problem data.

        Returns
        -------
        None.

        '''
        with open(in_file) as f:
            self.board_length = float(f.readline())
            for i,line in enumerate(f.readlines()):
                line_list = line.split(',')
                self.cut_sizes[i] = {'length':float(line_list[0]),
                                     'quant':int(line_list[1])}
        self.cut_sizes = pd.Dataframe(self.cut_sizes).transpose()
    
    
    def build_init_patterns(self):
        '''
        Builds the df mapping the patterns to the how many of each cut size

        Returns
        -------
        None.

        '''
        self.pattern_size_quant_df = pd.concat([self.pattern_size_quant_df,
                                                pd.DataFrame({pat:{cl:board_length//cl if i == pat else 0\
                                                                   for i,cl in enumerate(list(self.cut_sizes.length))}\
                                                              for pat in range(len(self.cut_sizes.index))}).transpose()],
                                               axis=0)



model = pyo.ConcreteModel()

cut_lengths = list(cut_sizes['length'])

pattern_size_quant_df = pd.DataFrame({pat:{cl:board_length//cl if i == pat else 0\
                                           for i,cl in enumerate(cut_lengths)}\
                                      for pat in range(len(cut_lengths))}).transpose()
patterns = list(pattern_size_quant_df.index)
    
# Decision variables x_p = num pattern p to use
model.x = pyo.Var(patterns,within = pyo.NonNegativeReals) #use NonNegativeInters for actual integer
#using Reals, because pyo doesn't give the dual variables for it.
# could potentially do a workaround by making another model with Reals
# but fix vars to the desired value, preprocess and do athe suffix
# see link: https://stackoverflow.com/questions/50468993/get-marginal-values-dual-for-constraints-in-pyomo-mip


# Objective function min total boards used
def obj_rule(model):
    return(sum(model.x[p] for p in patterns))
model.obj = pyo.Objective(rule = obj_rule)

# Demand constraint (patterns must satisfy each size quant constraint) sum(a_{ip}*x_p >=b_i)
def demand_con_rule(model,cut_length):
    return(sum(model.x[pattern]*pattern_size_quant_df.loc[pattern,cut_length] for pattern in patterns) >= int(cut_sizes.loc[cut_sizes['length']==cut_length,'quant']))
model.demand_con = pyo.Constraint(cut_lengths,
                                  rule=demand_con_rule)



##### Creates the suffixes (duals, reduced costs, and slack vars)
#tells pyomo to keep the dual vars (c_B B^{-1}) (dict keyed by constraints)
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
#tells pyomo to keep the variable reduced costs (c_{ij}-z_{ij} [obj row tableau coeffs])
# dict keyed by decision variables
model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
#tells pyomo to keep the slack var values
model.slack = pyo.Suffix(direction = pyo.Suffix.IMPORT)


solver = pyo.SolverFactory('glpk')
res = solver.solve(model) 
model.pprint()


####### Set up knapsack problem for column generation #######
dual_model = pyo.ConcreteModel()

# set is the cut length set; same as above

# Decision Variables x_j (num cuts of length j to include in new pattern) 
dual_model.y = pyo.Var(cut_lengths,within=pyo.NonNegativeIntegers)

def dual_obj_rule(dual_model):
    return(sum(dual_model.y[cl]*model.dual[model.demand_con[cl]] for cl in cut_lengths))
dual_model.obj = pyo.Objective(rule = dual_obj_rule,
                               sense=pyo.maximize)

def dual_weight_con_rule(model):
    return(sum(cl*dual_model.y[cl] for cl in cut_lengths) <= board_length)
    
dual_model.weight_con = pyo.Constraint(rule=dual_weight_con_rule)

dual_res = solver.solve(dual_model)
dual_model.pprint()

dual_obj = pyo.value(dual_model.obj)
if dual_obj >= 1:
    new_pattern = pd.DataFrame({len(pattern_size_quant_df.index):
                                {cl:int(pyo.value(dual_model.y[cl])) for cl in cut_lengths}}).transpose()

    
    
#### need to do the iteration of passing dual vars and making new pattern
#### need to establish the stopping condition (dual objective <= 1)