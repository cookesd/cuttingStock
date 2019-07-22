# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:23:29 2019

@author: cookesd
"""
from collections.abc import Iterable
import numbers
class cuttingStock(object):
    '''Class for a cutting stock object
    Requires the length of the stock and a dict of the required cuts and their quantities
    
    Has methods to get/set properties
    '''
    
    def __init__(self,boardLength = None, cutDict = None):
        self.boardLength = []
        self.cutDict = {}
        if boardLength is not None:
            self.add_board_length(boardLength)
        if cutDict is not None:
            self.add_cut(cutDict)
    
    def add_board_length(self,bl):
        if isinstance(bl,Iterable):
            for length in bl:
                if isinstance(length,numbers.Number):
                    self.boardLength.append(length)
        elif isinstance(bl,numbers.Number):
            if bl not in self.boardLength:
                self.boardLength.append(bl)
                
    def add_cut(self,cd):
        for key in cd:
            if key not in self.cutDict.keys:
                self.cutDict[key]=cd[key]
    
    def board_length(self):
        return self.boardLength
    
    def cutDict(self):
        return self.cutDict