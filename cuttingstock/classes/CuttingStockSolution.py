# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:28:58 2019

@author: dakar
"""
import pandas as pd
import numpy as np
import seaborn as sns
import plotnine as p9
import matplotlib.pyplot as plt
class CuttingStockSolution:
    
    def __init__(self,solution):
        self.sol = solution # the raw solution dataframe
        self.plot_df = self._make_plot_df() # the plotting dataframe
        self.cut_plot = self._make_p9_plot() # the stacked barplot
        
    def __repr__(self):
        temp = ''
        for pattern in self.sol:
            pattern_str = ''.join(['',
                                 '\n {} ({:.2f}) Cuts of Pattern {}:'.format(int(self.sol[pattern]['patternQuantity'][0]),
                                                                            float(self.sol[pattern]['patternQuantity'][1]),
                                                                            pattern[len(pattern)-1])])
            for cut in self.sol[pattern].index:
                if cut != 'patternQuantity' and cut != 'waste':
                    cut_str = '\n\t{} cuts of length {}'.format(int(self.sol[pattern][cut]),cut)
                    pattern_str = ''.join([pattern_str,cut_str])
            waste_str = '\n\twith {:.2f} units of waste.'.format(float(self.sol[pattern]['waste']))
            pattern_str = ''.join([pattern_str,waste_str])
            temp = '\n'.join([temp,pattern_str])
        return(temp)
                                 
        # return(self.printResult())
        
    
    def to_DataFrame(self):
        df = pd.DataFrame(self.sol)
        return(df)
    
    # def printResult(self):
    #     '''Prints the result in a tabular format to the console
        
    #     @param self.sol the pd.DataFrame of cut patterns with quantities of each cut size
    #     '''
    #     pat_str = ''
    #     for pattern in self.sol:
    #         print('\n',self.sol[pattern]['patternQuantity'][0],' (',np.round(self.sol[pattern]['patternQuantity'][1],2),') Cuts of Pattern ',pattern[len(pattern)-1],':',sep='')
    #         for cut in self.sol[pattern].index:
    #             if cut != 'patternQuantity' and cut != 'waste':
    #                 print(int(self.sol[pattern][cut]),'cuts of length',cut)
    #         print('with', np.round(self.sol[pattern]['waste'],2), 'units of waste')
    
    def _make_plot_df(self):
        plot_df = self.sol.drop(index='patternQuantity')
        plot_df['Cut Type'] = plot_df.index
        # melt the pattern columns to get x variable
        plot_df = pd.melt(plot_df,id_vars='Cut Type',
                          var_name='Pattern',value_name='Quantity')
        # Get the length for each cut (either from Cut Type if actual cut
        # or Quantity if a waste cut)
        plot_df['Length'] = plot_df.apply(lambda x: x['Cut Type']
                                          if x['Cut Type'] != 'waste'
                                          else x['Quantity'],axis='columns')
        # Fix Quantity column for waste cuts (quantity should be 1)
        plot_df.loc[plot_df['Cut Type']=='waste','Quantity'] = 1
        # Duplicates the rows based the on Quantity of each cut
        # (removes rows where the cut isn't present in that pattern)
        plot_df = plot_df.loc[plot_df.index.repeat(plot_df['Quantity'])]
        # Make Length Cat categorical variable to color bars by
        # Places bars bottom to top as largest cut to smallest and waste at top
        plot_df['Length Cat'] = pd.Categorical(plot_df['Cut Type'],
                                               categories=['waste'] +
                                               list(plot_df.loc[plot_df['Cut Type'] != 'waste',
                                                                'Length'].sort_values(ascending=True).unique()))
        # Make a column to display the quantity of the pattern and how much waste
        plot_df['Annotate'] = plot_df.apply(lambda x: '{} cuts of {}\nwith waste: {}'.format(self.sol.loc['patternQuantity',x['Pattern']][0],
                                                                                    x['Pattern'],x['Length'])
                                  if x['Length Cat']=='waste'
                                  else '',axis='columns')
        return(plot_df)
        
        
        
#     def plotResult(self):
#         '''Plots the result as a stacked bar graph using matplotlib.pyplot interface for pd.DataFrame
        
#         @param cut_dict the dictionary of cut patterns with quantities of each cut size
#         '''
        
# #        import seaborn as sns
# #        import matplotlib.pyplot as plt
#         epsilon = .005
        
#         cutsPerPattern = self.sol.drop(['patternQuantity','waste'],axis=0).sum()
#         colors = sns.color_palette("hls", len(cutsPerPattern.index)+1) #makes a palette of length for as many cut sizes we have
#         colorKeys = list(self.sol.drop('patternQuantity').index)
#         colorDict = {size:color for size,color in zip(colorKeys,colors)}
#         colorDict[0.0]='k'
#         maxNumCuts = int(max(cutsPerPattern))
#         rows = ['cut'+str(i+1) for i in range(maxNumCuts)]
#         cols = [col for col in self.sol.columns]
#         rows.append('waste')
        
#         plotDF = pd.DataFrame(index=rows,columns = cols) #makes empty df with appropriate row/cols & names
        
#         for pattern in self.sol.columns:
#             cutNum=1
#             patternDict = {}
#             for index in self.sol.drop(['patternQuantity','waste']).index:
#                 #accounts for duplicate cuts of the same size & skips sizes with zero cuts
#                 if self.sol[pattern][index]>0:
#                     for i in range(int(self.sol[pattern][index])):
#                         patternDict['cut'+str(cutNum)]=float(index)
#                         cutNum+=1
#             #fills remaining cuts with zero lengths
#             if cutNum <= maxNumCuts:
#                 remainingCuts = maxNumCuts+1-cutNum
#                 for i in range(int(remainingCuts)):
#                     patternDict['cut'+str(cutNum+i)]=0
#             patternDict['waste']=self.sol[pattern]['waste'] #adds the waste row
#             patternSeries = pd.Series(patternDict) #makes dict a series
#             plotDF[pattern] = patternSeries #adds the series for the pattern in the appropriate col in self.sol
        
#         plotDF = plotDF.transpose() #transposes plotDF to make the stacked bar plot
#         graphHeight = sum(plotDF.loc['Pattern1',:])*1.1
#         #plots cut patterns using stacked bargraphs
# #        ax =plotDF.plot(kind='bar',stacked=True,edgecolor='k',legend=False) #plots stacked bar graph
# #        for p in ax.patches:
# #            #plots cut size in each bar if height > 0.0
# #            if p.get_height()>epsilon:
# #                ax.annotate(str(p.get_height()), (p.get_x()+.25*p.get_width(),p.get_y()+.5*p.get_height()))
# #        
        
#         # plots each cut as a bar in a stacked bar chart
#         # The bar colors correspond to the cut length (i.e., all cuts of size x are the same color)
#         bottomVals = 0
#         for i in range(len(plotDF.columns)):
#             if i > 0 and plotDF.columns[i] != 'waste':
# #                plotDF[plotDF.columns[i]].plot(kind='bar',edgecolor='k',color = [colorDict[plotDF[plotDF.columns[i]][j]] for j in plotDF[plotDF.columns[i]].index])
#                 ax = plotDF[plotDF.columns[i]].plot(kind='bar',edgecolor='k',color = [colorDict[plotDF[plotDF.columns[i]][j]] for j in plotDF[plotDF.columns[i]].index],bottom = bottomVals)
#                 for p in ax.patches:
#                     if p.get_height() > epsilon:
#                         ax.annotate(str(p.get_height()), (p.get_x()+.25*p.get_width(),p.get_y()+.5*p.get_height()))
#                 bottomVals += plotDF[plotDF.columns[i]]
#             elif plotDF.columns[i]== 'waste':
# #                plotDF[plotDF.columns[i]].plot(kind='bar',edgecolor='k',color = colorDict['waste'])
#                 ax = plotDF[plotDF.columns[i]].plot(kind='bar',edgecolor='k',color = colorDict['waste'],bottom = bottomVals)
#                 for p in ax.patches:
#                     if p.get_height() > epsilon:
#                         ax.annotate(str(p.get_height()), (p.get_x()+.25*p.get_width(),p.get_y()+.5*p.get_height()))
#             else:
#                 ax = plotDF[plotDF.columns[i]].plot(kind='bar',edgecolor='k',color = [colorDict[plotDF[plotDF.columns[i]][j]] for j in plotDF[plotDF.columns[i]].index],ylim = (0,graphHeight))
#                 for p in ax.patches:
#                     if p.get_height() > epsilon:
#                         ax.annotate(str(p.get_height()), (p.get_x()+.25*p.get_width(),p.get_y()+.5*p.get_height()))
#                     bottomVals = list(plotDF[plotDF.columns[i]])
    
    def _make_p9_plot(self):
        '''
        Make ggplot2 style stacked barplot of cutting patterns annotated with
        waste and pattern quantities.
        
        Stacked bars are colored based on the cut type (either the cut length
        or waste).

        Returns
        -------
        g : plotnine ggplot.

        '''
        # self.make_plot_df()
        g = (p9.ggplot(mapping=p9.aes(x='Pattern',y='Length',fill='Length Cat'),
                       data=self.plot_df) +
             p9.geom_bar(position='stack',stat='identity',color='black') +
             p9.scale_fill_brewer(type='qual',palette=2,name='Cut Type') +
             p9.geom_text(mapping=p9.aes(y='Length',label='Annotate'),
                          position='stack') +
             p9.ggtitle('Pattern Cuts'))
        return(g)
             