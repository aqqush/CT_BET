#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:22:53 2017

@author: m131199
"""
import os
import nibabel as nb
import numpy as np

#def arrangeData(images, labels, fold, dtype):
#
#    [img_rows,img_cols,numImgs] = images.shape
##    a=np.array([16,32,64,128,256])
##    m = numImgs/(a*1.0)
##    fold = int(a[1+np.where((m>=1) & (m<2))[0]])
#    pad = np.zeros((img_rows,img_cols,fold-numImgs)) 
#    images = np.concatenate((images,pad-1024), axis=2)
#    labels = np.concatenate((labels,pad), axis=2)
#    numI = (img_rows/(2*fold))*(img_cols/(2*fold))
#    images= images.reshape(numI,2*fold,2*fold,fold,1).astype(dtype)
#    labels= labels.reshape(numI,2*fold,2*fold,fold,1).astype(dtype)
#    return images,labels,numImgs

def arrangeData(images, labels, fold, dtype):

    [img_rows,img_cols,numImgs] = images.shape
#    a=np.array([16,32,64,128,256])
#    m = numImgs/(a*1.0)
#    fold = int(a[1+np.where((m>=1) & (m<2))[0]])
    pad = np.zeros((img_rows,img_cols,fold-numImgs)) 
    images = np.concatenate((images,pad-1024), axis=2)
    labels = np.concatenate((labels,pad), axis=2)
    images= images.reshape(1,img_rows,img_cols,fold,1).astype('float32')
    labels= labels.reshape(1,img_rows,img_cols,fold,1).astype('uint8')
    return images,labels
  
  
  