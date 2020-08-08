# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:08:28 2019

@author: dakar

Reference for adding plot to OutputPage
https://datatofish.com/matplotlib-charts-tkinter-gui/

https://docs.python.org/3/library/tk.html
"""


import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DEFAULT_TEXT = 'No board lengths specified'

class CuttingStock():
    def __init__(self,*args,**kwargs):
        #instantiates Input Window
        self.in_window = InputWindow(self)
        self.out_window = None
        self.cuts = pd.DataFrame(columns = ['cut_length','num_cuts'])
        self.stock_length = None
        
        
        
        # optimization attributes
        self.cB = None
        self.b = None
        self.B = None
        self.Binv = None
        self.res = None # empty var for the result once optimized (should be CuttingStockResult object)
        self.dual_vars = None
        self.enteringCol = None
        self.benefit = None
        self.epsilon = .005
        
        # begin optimization
        self.in_window.mainloop()
        
        
    def add_cut(self,cut_length = 0, num_cuts = 0):
        self.cuts = self.cuts.append(pd.Series({'cut_length':int(cut_length),'num_cuts':int(num_cuts)},
                                               name=len(self.cuts)))
    
    def set_stock_length(self,stock_length):
        self.stock_length = stock_length
        
    def read_data(self,file_path):
        with open(file_path) as f:
            self.stock_length = int(f.readline())
            for line in f.readlines():
                line=line.split(',')
                self.add_cut(line[0],line[1])
    
    def init_vals(self):
        Bdiag = np.array([np.floor(self.stock_length/self.cuts['cut_length'][i]) for i in self.cuts.index])
        Bdim = len(Bdiag)
        self.b = np.array(self.cuts['num_cuts'],ndmin=2).T # sets the RHS values as column vector
        self.B = np.diag(Bdiag) # sets the initial basis
        self.cB = np.array([1]*Bdim) # sets the objective coefficients
        self.Binv = np.linalg.inv(self.B)
        
        # begins the initial iteration
        self.dual_vars = self.calc_dual_vars()
        self.benefit,self.enteringCol = self.knapsack(self.dual_vars,self.cuts['cut_length'],self.stock_length,[0]*len(self.b))
        self.enteringCol = np.reshape(self.enteringCol,(len(self.enteringCol),1)) #ensures this is column vector
        
            
        
    def optimize(self):
        self.in_window.frames[InputPage].readback_lbl.configure(text='Optimizing!!!')
        self.in_window.destroy()
        self.init_vals() # initializes the tableau attributes for optimization
        while  self.benefit - 1 > self.epsilon: #not optimal
            self.unbounded, self.lv, self.bbar, self.abar = self.findLV()
            if not self.unbounded: # not unbounded
                self.iterate()
            else: # unbounded
                
                self.declare_unbounded()
            # print('\a')
            # print('{:*^10}')
            # self.print_iteration()
        
        # do another iteration to update to final iteration
        self.unbounded, self.lv, self.bbar, self.abar = self.findLV()
        if not self.unbounded:
            self.iterate()
        self.res = CuttingStockResult(self)
        
        self.display_result()
        
    def display_result(self):
        self.out_window = OutputWindow(self)
        self.out_window.mainloop()
    

    def print_iteration(self):
        
        print('B\n',self.B)
        print('bbar\n',self.bbar)
        print('cb\n',self.cB)
        print('lv\n',self.lv)
        print('benefit\n',self.benefit,'\n')
        
    def declare_unbounded(self):
        print('The problem is unbounded')
        self.benefit = 0

            
    def iterate(self):
        self.Binv = self.updateBinv()
        self.B = np.linalg.inv(self.Binv)
        self.dual_vars = self.calc_dual_vars()
        self.benefit, self.enteringCol = self.knapsack(self.dual_vars,self.cuts['cut_length'],self.stock_length,[0]*len(self.b))
        self.enteringCol = np.reshape(self.enteringCol, (len(self.enteringCol),1)) # ensures this is a column vector
        # self.iterate_callback()
        
    def iterate_callback(self):
        print('{:*^21}'.format('New iteration'))
        print('dual_vars: {}'.format(self.dual_vars))
        print('benefit: {}'.format(self.benefit))
        print('entering column: {}'.format(self.enteringCol))
        
    
    def knapsack(self,values, weights, capacity,solVect = None):
        '''Solves the unbounded knapsack problem using dynamic programming (recursion).
        The unbounded knapsack problem here tries to maximize the value (dual variable of the entering cut pattern)
        subject to the capacity constraints (the board cuts cannot exceed the board length).
        This new pattern will enter our basis if the value (dual var) is greater than 1,
        Otherwise, it will not improve the objective to the linear program.
        
        @param values (iterable of floats) : knapsack obj function coefficients (the current dual variables for the linear programming solution (c_{B}B^{-1}))
        @param weights (iterable of floats) : knapsack constraint coefficients for each cut (the length of the desired cuts)
        @param capacity (float) : the knapsack capacity (length of the board)
        @param solVect {optional} (iterable of length number of cuts) : should be a list of zeros initially;
        used for recursively calling knapsack; if no value specified, then automatically sets to list of zeros
        If a vector is applied, it is a starting knapsack solution
        
        returns finalSol : the solution to the knapsack ()
        '''
        solMat = np.array([solVect]*len(values))
        sol = [0]*len(values) #solution to the subproblem (capacity-values[i]) after adding item i to knapsack
        largerSol = [0]*len(values) #solution to subproblem plus adding item i
        finalSol = None
        # finds the max value for the subproblem with capacity (after removing capacity for that item)
        for i in range(len(values)):
            if weights[i] <= capacity:
                # print('len_vals: {} w: {} c: {}'.format(len(values),weights[i],capacity))
                newCap = capacity-weights[i]
                solMat[i][i] +=1 #adding 1 cut of size "weight" to the solution matrix
                # print('{:*^21}'.format('New Iteration'))
                # print(solMat)
                sol[i],solMat[i] = self.knapsack(values, weights, newCap,solMat[i]) #calls knapsack with the updated capacity after the new cut has been added
                
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
    
    def findLV(self):
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
        tol = -1.0E-12
        largeNum = max(self.b)**2 #this should be big enough, need to find better value
        bbar = np.matmul(self.Binv,self.b)
        abar = np.matmul(self.Binv,self.enteringCol)
        # print('bbar: {} \n abar: {}'.format([type(i) for i in bbar],[type(j) for j in abar]))
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

    def updateBinv(self):
        '''
        Updates the B^{-1} matrix with the new col (abar) in place of the leaving column (same column as lvrow since square matrix)
        
        @param Binv : the B^{-1} matrix from the previous solution
        @param abar : the column for the entering basic variable when premultiplied by the current B^{-1} matrix
        (B^{-1}*a_i for new basic variable i)
        @para lvrow : the row corresponding to the leaving variable found using the ratio test (b_j / a_{ij}). In case of ties, lowest index chosen
        
        returns newBinv : the updated B^{-1} matrix
        '''
        matDim = len(self.Binv)
        eMat = np.identity(matDim) #identity matrix with same size as Binv
        newCol = -self.abar/self.abar[self.lv] #the lvrowth column (r) should be -abar_ik/abar_rk with (r,r) element = 1/abar_rk
        newCol[self.lv] = 1/self.abar[self.lv]
        # print('entering column\n', newCol)
        eMat[:,self.lv] = np.reshape(newCol,(1,matDim)) # places newCol into the lvrow column
        
        newBinv = np.matmul(eMat,self.Binv) #updates B^{-1} using E*B^{-1}
        return(newBinv)
    
    def calc_dual_vars(self):
        '''Uses matrix multiplication to calculate c_B*B{^-1} (dual variables w for the current solution)
        
        @param cB : the cost coefficients of the basic variables
        @param Binv : the B^{-1} matrix for the current basis
        
        returns cBBinv : the current dual variables (w = c_B*B^{-1})
        '''
        cBBinv = np.matmul(self.cB,self.Binv)
        return(cBBinv)
    
    
class CuttingStockResult():
    def __init__(self,cs_problem):
        self.cs = cs_problem
        
        cutDict = {}
        for colNum in range(len(self.cs.bbar)):
            dictKey = 'Pattern' + str(colNum + 1)
            cutDict[dictKey] = {'patternQuantity':(np.ceil(self.cs.bbar[colNum]),self.cs.bbar[colNum])}
            pattern = [(self.cs.B[j][colNum],self.cs.cuts['cut_length'][j]) for j in self.cs.cuts.index]
            waste = self.cs.stock_length - sum([i*j for (i,j) in pattern])
            for cut in pattern:
                cutDict[dictKey][cut[1]] = cut[0]
            cutDict[dictKey]['waste'] = waste
        self.cut_df = pd.DataFrame(cutDict)
        self.result_df = self.cleanResult()
        self.plot_df = self.makePlotDF()
        self.fig = self.plotResult();
        # plt.plot(self.f)
        
    def __str__(self):
        s = ''
        _ = ' '
        
        for pattern in self.result_df:
            s+= _.join(['\n',str(self.result_df[pattern]['patternQuantity'][0]),' (',str(np.round(self.result_df[pattern]['patternQuantity'][1],2)),') Cuts of Pattern ',pattern[len(pattern)-1],':'])
            # print('\n',self.result_df[pattern]['patternQuantity'][0],' (',np.round(self.result_df[pattern]['patternQuantity'][1],2),') Cuts of Pattern ',pattern[len(pattern)-1],':',sep='')
            for cut in self.result_df[pattern].index:
                if cut != 'patternQuantity' and cut != 'waste':
                    s += _.join(['\n',str(int(self.result_df[pattern][cut])),'cuts of length',str(cut)])
                    # print(int(self.result_df[pattern][cut]),'cuts of length',cut)
            s+= _.join(['\n','with', str(np.round(self.result_df[pattern]['waste'],2)), 'units of waste','\n'])
            # print('with', np.round(self.result_df[pattern]['waste'],2), 'units of waste')
        return(s)
        
    def cleanResult(self):
        '''Cleans the resulting DF for printing and plotting for user
        makes the useful pattern quantity an integer and rounds the actual quantity to 3 decimals
        makes each cut quantity an integer rounded to the nearest integer (fixes computer calculation rounding)
        
        @param resultDF (pd.DataFrame) : the result DataFrame from the cutting stock problem
        returns the cleaned pd.DataFrame'''
        
        clean = self.cut_df.copy()
        for p in clean.columns:
            for i in clean[p].index:
                if i == 'patternQuantity':
                    clean[p][i] = (np.array(clean[p][i][0][0]),np.round(clean[p][i][1][0],3)) #extra 0 gets the value not the array
                    # clean[p][i] = (np.array(int(clean[p][i][0])),np.round(clean[p][i][1],3))
                else:
                    clean[p][i] = int(np.round(clean[p][i],0))
        return(clean)
        
        
    # def printResult(self):
    #     '''Prints the result in a tabular format to the console
        
    #     @param self.result_df the pd.DataFrame of cut patterns with quantities of each cut size
    #     '''
        
    #     for pattern in self.result_df:
    #         print('\n',self.result_df[pattern]['patternQuantity'][0],' (',np.round(self.result_df[pattern]['patternQuantity'][1],2),') Cuts of Pattern ',pattern[len(pattern)-1],':',sep='')
    #         for cut in self.result_df[pattern].index:
    #             if cut != 'patternQuantity' and cut != 'waste':
    #                 print(int(self.result_df[pattern][cut]),'cuts of length',cut)
    #         print('with', np.round(self.result_df[pattern]['waste'],2), 'units of waste')
    
    def makePlotDF(self):
        '''Plots the result as a stacked bar graph using matplotlib.pyplot interface for pd.DataFrame
        
        @param cut_dict the dictionary of cut patterns with quantities of each cut size
        '''
        cutsPerPattern = self.result_df.drop(['patternQuantity','waste'],axis=0).sum()
        maxNumCuts = int(max(cutsPerPattern))
        rows = ['cut'+str(i+1) for i in range(maxNumCuts)]
        cols = [col for col in self.result_df.columns]
        rows.append('waste')
        
        plotDF = pd.DataFrame(index=rows,columns = cols) #makes empty df with appropriate row/cols & names
        
        for pattern in self.result_df.columns:
            cutNum=1
            patternDict = {}
            for index in self.result_df.drop(['patternQuantity','waste']).index:
                #accounts for duplicate cuts of the same size & skips sizes with zero cuts
                if self.result_df[pattern][index]>0:
                    for i in range(int(self.result_df[pattern][index])):
                        patternDict['cut'+str(cutNum)]=float(index)
                        cutNum+=1
            #fills remaining cuts with zero lengths
            if cutNum <= maxNumCuts:
                remainingCuts = maxNumCuts+1-cutNum
                for i in range(int(remainingCuts)):
                    patternDict['cut'+str(cutNum+i)]=0
            patternDict['waste']=self.result_df[pattern]['waste'] #adds the waste row
            patternSeries = pd.Series(patternDict) #makes dict a series
            plotDF[pattern] = patternSeries #adds the series for the pattern in the appropriate col in self.result_df
        
        plotDF = plotDF.transpose() #transposes plotDF to make the stacked bar plot
        return(plotDF)
        #plots cut patterns using stacked bargraphs
        
    # def plotResult(self):
    #     ax =self.plot_df.plot(kind='bar',stacked=True,edgecolor='k',legend=False)#,colormap='Greens') #plots stacked bar graph
    #     for p in ax.patches:
    #         #plots cut size in each bar if height > 0.0
    #         if p.get_height()>self.cs.epsilon:
    #             ax.annotate(str(p.get_height()), (p.get_x()+.125*p.get_width(),p.get_y()+.5*p.get_height()))
    
    # def plotResult2(self):
    #     colors = mcolors.BASE_COLORS.keys()
    #     vals = np.unique(self.plot_df.values)
    #     c_dict = {v:c for v,c in zip(vals,colors)}
    #     for i in range(len(self.plot_df.columns)):
    #         plt.bar(self.plot_df.index,
    #                 self.plot_df[self.plot_df.columns[i]],
    #                 color=[c_dict[v] for v in self.plot_df[self.plot_df.columns[i]]],
    #                 bottom = self.plot_df.iloc[:,:i].sum(1),
    #                 edgecolor='k')
    #     # for c,v in c_dict.items():
    #     # blank Artists for legend
    #     legend_elements = [Line2D([0],[0],color=c,label=v) for v,c in c_dict.items() if int(v) > 0]
    #     plt.legend(handles=legend_elements,
    #                loc='center left',
    #                bbox_to_anchor=(1,0.5))
    #     plt.title('Optimal Patterns')
    #     plt.xlabel('Patterns')
    #     plt.ylabel('Cut Size')
        
    def plotResult(self):
        '''Makes a stacked barplot for the cutting patterns created
        The bars are color-coded by the cut length
        Waste cuts are all colored the same'''
        
        mpl.style.use('seaborn') #the seaborn style is nices
        
        # make color dictionary for plots
        vals = np.unique(self.plot_df.values)
        waste_col = 'C'+str(len(vals)+2)
        c_dict = {v:'C'+str(i+1) if v in list(self.cs.cuts['cut_length']) else waste_col for i,v in enumerate(vals)}
        
        #make stacked bar plots
        f = plt.figure(figsize=(8,5))
        ax = f.add_subplot(121)
        for i in range(len(self.plot_df.columns)):
            ax.bar(self.plot_df.index,
                    self.plot_df[self.plot_df.columns[i]],
                    color=[c_dict[v] for v in self.plot_df[self.plot_df.columns[i]]],
                    bottom = self.plot_df.iloc[:,:i].sum(1),
                    edgecolor='k')
                    # figure=f)
        
        # blank Artists for legend
        legend_elements = [Line2D([0],[0],color=c,label=v,figure=f) for v,c in c_dict.items() if v in self.cs.cuts['cut_length'].values]
        legend_elements.append(Line2D([0],[0],color=waste_col,label='waste',figure=f))#figure=f))
        plt.legend(handles=legend_elements,
                   loc='best',#'center left',
                   bbox_to_anchor=(1,0.5))
        
        # puts title and axes labels
        plt.title('Optimal Patterns')
        plt.xlabel('Patterns')
        plt.ylabel('Cut Size');
        return(f)


class InputWindow(tk.Tk):
    def __init__(self,problem,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs) #runs the Tk __init__
        self.title('Input Window')
        
        container = tk.Frame(self) #initializes from for pages to go in
        container.pack(side='top',fill='both',expand=True)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)
        
        self.problem = problem
        self.frames = {}
        for i,F in enumerate([InputPage]):
            frame = F(container,self)
            self.frames[F] = frame
            frame.grid(row=3*i,column=3*i,sticky='ns')
            

            
            
class InputPage(tk.Frame):
    def __init__(self,parent,controller,**kwargs):
        tk.Frame.__init__(self,parent)
        
        self.controller = controller
        
        self.len_lbl = tk.Label(parent,text='Enter cut length (inches):')
        self.cut_lbl = tk.Label(parent,text='Enter the number of cuts:')
        self.file_lbl = tk.Label(parent,text='Enter the absolute path for entry file:')
        self.stock_size_lbl = tk.Label(parent,text = 'Enter the stock size (inches):')

        #Set label locations
        self.stock_size_lbl.grid(column = 0,row = 0)
        self.len_lbl.grid(column=0,row=1)
        self.cut_lbl.grid(column=0,row=2)
        self.file_lbl.grid(column=0,row=3)
        

        # make entry boxes
        self.len_entry = tk.Entry(parent)
        self.cut_entry = tk.Entry(parent)
        self.stock_size_entry = tk.Entry(parent)
        self.file_entry = tk.Entry(parent)
        self.file_entry.configure(state='disabled')
        # set entry locations
        self.stock_size_entry.grid(column=1,row=0)
        self.len_entry.grid(column=1,row=1)
        self.cut_entry.grid(column=1,row=2)
        self.file_entry.grid(column=1,row=3)
        
        
        
        
        self.readback_lbl = tk.Label(parent,text=DEFAULT_TEXT) #makes space for printout
        self.readback_lbl.grid(row=6,columnspan=2) #places the Label so it displays
        
        # initialize the entry boxes
        self.initialize_entries()
        
        
        
        # make buttons
        self.enter_new_button = tk.Button(parent,text='Enter & Add New',command=self.enter_new)
        self.enter_finished_button = tk.Button(parent,text='Enter & Finished',command=self.enter_finished,state='disabled')
        # self.edit_button = tk.Button(parent,text='Edit',command=self.edit)
        self.optimize_button = tk.Button(parent,text='Optimize',command=self.optimize,state='disabled')
        self.enter_stock_button = tk.Button(parent,text='Enter Stock Length',command=self.enter_stock)
        self.enter_path_button = tk.Button(parent,text='Enter File Path',command=self.enter_path)
        # place buttons
        self.enter_new_button.grid(column=0,row=4)
        self.enter_finished_button.grid(column=1,row=4)
        # self.edit_button.grid(column=2,row=4)
        self.optimize_button.grid(column=2,row=4)
        self.enter_stock_button.grid(column=2,row=0)
        self.enter_path_button.grid(column=2,row=3)
    
    def initialize_entries(self):
        # determine the amount of text in the box
        len_len = len(self.len_entry.get())
        cut_len = len(self.cut_entry.get())
        stock_size_len = len(self.stock_size_entry.get())
        
        # delete current information and set to default
        self.len_entry.delete(0,len_len)
        self.len_entry.insert(index = 0, string = '0')
        self.cut_entry.delete(0,cut_len)
        self.cut_entry.insert(index = 0, string = '0')
        self.stock_size_entry.delete(0,stock_size_len)
        self.stock_size_entry.insert(index = 0, string = '0')
    
    def enter_stock(self):
        
        stock_text = 'The stock is {} inches.'.format(self.stock_size_entry.get())
        
        if self.controller.problem.stock_length:
            # There is a stock length set
            if self.controller.problem.stock_length != self.stock_size_entry.get():
                # The current stock length is different
                self.readback_lbl.configure(text = stock_text + ' Changed from {}.'.format(self.controller.problem.stock_length) + self.readback_lbl.cget('text').split('.')[-1])
                self.controller.problem.set_stock_length(self.stock_size_entry.get())
        else:
            self.controller.problem.set_stock_length(self.stock_size_entry.get())
            # There is no stock length set
            if self.readback_lbl.cget('text') == DEFAULT_TEXT:
                # Only the default text is shown
                self.readback_lbl.configure(text = stock_text)
            else:
                self.readback_lbl.configure(text = stock_text + '\n' + self.readback_lbl.cget('text'))
        
        self.check_finish()
    
    def enter_path(self):
        in_file = fd.askopenfilename()
        if in_file:
            self.file_entry.configure(state='normal') #make the entrybox selectable
            len_file = len(self.file_entry.get()) #number of characters currently in the entrybox
            self.file_entry.delete(0,len_file) #delete current info
            self.file_entry.insert(index=0,string=in_file) #insert file path to entrybox
        self.readback_lbl.configure(text = 'Reading data from: {}'.format(in_file))
        self.controller.problem.read_data(in_file)
        self.readback_lbl.configure(text = self.controller.problem.cuts)
        self.check_finish()
        
        
    def enter_new(self):
        self.controller.problem.add_cut(self.len_entry.get(),
                                    self.cut_entry.get())
        current_cut = 'Cut {}: {} cuts of length {}' .format(len(self.controller.problem.cuts),
                                                             self.cut_entry.get(),
                                                             self.len_entry.get())
        if self.readback_lbl.cget('text') == DEFAULT_TEXT:
            self.readback_lbl.configure(text = current_cut)
        else:
            self.readback_lbl.configure(text = self.readback_lbl.cget('text') + '\n' + current_cut)
        self.initialize_entries()
        
        self.check_finish()
        
        
    def enter_finished(self):
        finished_text = 'You are done entering cuts.'
        self.enter_new()
        self.readback_lbl.configure(text = finished_text + '\n' + self.readback_lbl.cget('text'))
        self.enter_new_button.configure(state='disabled')
        self.enter_finished_button.configure(state='disabled')
        self.enter_stock_button.configure(state='disabled')
        self.optimize_button.configure(state='normal')
        
        
    def check_finish(self):
        if self.controller.problem.stock_length and not self.controller.problem.cuts.empty:
            self.enter_finished_button.configure(state='normal')
            self.optimize_button.configure(stat='normal')
            
    # def edit(self):
    #     self.readback_lbl.configure(text = 'You pressed edit!')

    def optimize(self):
        # self.readback_lbl.configure(text = 'You pressed optimize!')
        self.controller.problem.optimize()
        
class OutputWindow(tk.Tk):
    def __init__(self,problem,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs) #runs the Tk __init__
        self.title('Output Window') #labels the title bar
        
        container = tk.Frame(self) #initializes from for pages to go in
        container.grid(column=0,row=0,sticky='nsew')
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)
        
        self.problem = problem
        self.frames = {}
        for i,F in enumerate([OutputPage]):
            frame = F(container,self)
            self.frames[F] = frame
            # frame.pack()
            frame.grid(row=3*i,column=3*i,sticky='nsew')
            
            
class OutputPage(tk.Frame):
    def __init__(self,parent,controller,**kwargs):
        tk.Frame.__init__(self,parent)
        
        self.controller = controller
        
        self.output_lbl = tk.Label(parent,text=self.controller.problem.res)
        self.output_lbl.grid(column=0,row=0,padx=1)
        # self.output_lbl.pack()
        
        chart_type = FigureCanvasTkAgg(self.controller.problem.res.fig,self.controller)
        chart_type.get_tk_widget().grid(column=1,row=0)
        # self.canvas = FigureCanvasTkAgg()



prob = CuttingStock()  
# prob.in_window.mainloop()    
# window = InputWindow()
# window.mainloop()