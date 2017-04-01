#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:54:38 2017

@author: guangwei
"""
import numpy as np
import matplotlib.pyplot as plt
from FunParam import Param

#%matplotlib qt

def  VisualizeNet(wCurrent, currentError, wTarget, Param):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    
    ax1.imshow(wCurrent[-1], extent=[1, Param.numNode[1], 1, Param.numNode[0]])
    ax1.set_title('Current')

    ax2.imshow(wTarget[-1], extent=[1, Param.numNode[1], 1, Param.numNode[0]])
    ax2.set_title('Target')

    #ax3.bar([1,2,3,4], currentError )
    ax3.bar(list(range(1, Param.numNode[1]+1)), currentError )
    ax3.set_title('Error')
    
    return

""" 
#test
VisualizeNet(weightTrain, nodeError[0].transpose(), weightTarget)
"""