# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 19:24:48 2019

@author: dakar
"""



class CuttingStock:
    from .solving import solve #imports the solve method of the class
    # I could set class attributes here using <attribute>=<value>
    # All instances of the class would have that value
    cut_factory = dict
    def __init__(self,cuts = None,boardLength = None):
        '''
        Creates a CuttingStock problem
        @param cuts will be made into a dict of dicts
            the key is the cut-length the value is a dict of the attributes
            one of the attributes must be the required quantity
            The input can either be this dict of dicts or an iterable containing
            the cut length and the required quantity only
        @param boardLength is either int or float
        '''
        self.cuts = self.cut_factory() #makes it whatever type the cut_factory is set to. Easier to fix
        if cuts:
            self.add_cuts(cuts)
        self.set_boardLength(boardLength) #may set it to None if not passed in
        
    def __str__(self):
        '''Sets the string representation of the object for use with print function'''
        return 'This cutting stock problem has {} required cut lengths using a {} inch board.\n The cuts are:\n {}'.format(len(self.cuts),
                                               self.boardLength,
                                               self.cuts)
        
    def __iter__(self):
        
        return iter(self.cuts)
    
#    def add_cuts(self,cuts):
#        try:
#            iter(cuts)
#            for cut in cuts:
#                if isinstance(cut,Cut):
#                    self.cuts.append(cut)
#                else:
#                    raise TypeError('cuts must be an instance of type Cut,iter')
#        except TypeError: #if not iterable object, assumings it's a single object
#            print(iter(cuts))
#            if isinstance(cuts,Cut):
#                self.cuts.append(cuts)
#            else:
#                raise TypeError('cuts must be an instance of type Cut,noniter')
                
    def add_cuts(self,cuts):
        import pandas as pd
        from collections.abc import Iterable
        # cuts is dict or pd.DataFrmae
        if isinstance(cuts,(dict,pd.DataFrame)):
            if isinstance(cuts,pd.DataFrame):
                cuts = cuts.to_dict()
            
        # cuts is pd.Series
        elif isinstance(cuts,pd.Series):
            cuts = cuts.to_dict()
            cuts = {key:{'reqQuant':cuts[key]} for key in cuts.keys()}
        # cuts is list/tuple of lists/tuples
        elif isinstance(cuts,Iterable):
            try:
                isinstance(cuts,(list,tuple))
                cuts = {cuts[i][0]:{'reqQuant':cuts[i][1]} for i in cuts}
            except TypeError:
                print('cuts must be a list or tuple of lists or tuples')
        else:
            raise TypeError('cuts must be dict, pd.DataFrame, pd.Series, or list/tuple of lists/tuples')
        
        #add the cuts to the cut-dict if not already included
        for key in cuts.keys():
                try:
                    isinstance(key,(int,float))
                #add cut to cut-dict if not already there
                    try:
                        key not in self.cuts.keys()
                        self.cuts[key] = {'reqQuant':cuts[key]}
                    #if cut already in cut-dict, raise key error
                    except KeyError:
                        print('The Cutting Stock object already has cut with length {}. Use update_cuts instead'.format(key))
                except TypeError:
                    print('The cut length must be int or float') 
                            
    def update_cuts(self,updateDict):
        '''Updates the attributes for a cut
        cutDict is a dict of dicts keyed by the cut length
        value is a dict keyed by the desired attributes to update (must already be attributes)
        those values are the value to change'''
        for key in updateDict:
            try:
                key in self.cuts.keys()
                for subkey in updateDict[key].keys():
                    self.cuts[key][subkey] = updateDict[key][subkey]
                    
            except KeyError:
                print('The Cutting Stock object does not have a cut with length {}.'.format(key))
                
    def remove_cuts(self,cutList):
        '''Updates the attributes for a cut
        cutDict is a dict of dicts keyed by the cut length
        value is a dict keyed by the desired attributes to update (must already be attributes)
        those values are the value to change'''
        if isinstance (cutList,(int,float)):
            cutList = list((cutList,))
        for cut in cutList:
            try:
                cut in self.cuts.keys()
                del self.cuts[cut]
            except KeyError:
                print('The Cutting Stock object does not have a cut with length {}.'.format(cut))
    
    def set_boardLength(self,boardLength):
        if boardLength:
            try:
                isinstance(boardLength,(int,float))
                self.boardLength = boardLength

            except TypeError:
                print('boardLength must be an int or float')
        else:
            self.boardLength = None #sets to None. Only for initial instantiation
        
    def copy(self):
        import copy
        return CuttingStock(copy.deepcopy(self.cuts),self.boardLength)
            
        
#class Cut:
#    factory = dict
#    def __init__(self,cutLength,reqQuantity,*attr):
#        self.cutLength = cutLength
#        self.reqQuantity = reqQuantity
#    def __str__(self):
#        return '{} length {} cuts'.format(self.reqQuantity,self.cutLength)
#    def __repr__(self):
#        return '[cut length: {}, required quantity {}]'.format(self.cutLength,self.reqQuantity)
    
##### Testing creating and using the objects #####
#cut1 = Cut(5,12)
#cut2 = Cut(24,6)

#cutDict = {0: {'reqQuant': 5, 'attr2': 1, 'attr3': 'sdfs'},
#           1: {'reqQuant': 10, 'attr2': 47}}
#cs1 = CuttingStock(cutDict,10)
#
#### testing methods ###
#updateCutDict = {1:{'attr2':13,'attr3':12},0:{'attr3':50}}
#addCutDict = {2:{'reqQuant':15}}
#delCutDict = 2
#bl = 240
#
#
#cs2 = cs1.copy()
#
#print(cs1.cuts)
#print(cs1.boardLength)
#
#
#cs1.update_cuts(updateCutDict)
#print(cs1.cuts)
#
#cs1.add_cuts(addCutDict)
#print(cs1.cuts)
#
#cs1.remove_cuts(delCutDict)
#print(cs1.cuts)
#
#cs1.set_boardLength(bl)
#print(cs1.cuts)
#print(cs1.boardLength)
#
#print(cs2.cuts)
#print(cs2.boardLength)