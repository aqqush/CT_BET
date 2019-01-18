#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:28:01 2017

@author: m131199
"""

import os
import numpy as np
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.layers import (Input, concatenate, Conv2D, 
                          MaxPooling2D, Conv2DTranspose, Activation, 
                          Dropout, Flatten, Reshape, Cropping3D,
                          Dense, ZeroPadding2D, AveragePooling2D,
                          GlobalAveragePooling2D, GlobalMaxPooling2D,
                          BatchNormalization)
from keras.utils import np_utils
from deepModels import Unet, Unet3D
from sklearn import metrics
import nibabel as nb
from sklearn.model_selection import KFold
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
import SimpleITK as sitk
from  scipy.ndimage.interpolation import zoom as interp3D
from load3Ddata import arrangeData as arrange3Ddata

class Unet_CT_SS(object):

    def __init__(self, root_folder='', code_folder='',image_folder = 'image_data',
                 mask_folder = 'mask_data',save_folder='', training_folder='',
                 test_folder='',pred_folder='',testingMode=False, savePredMask='False',
                 testLabelFlag=True,testMetricFlag=False, dataAugmentation = False,
                 includeNC=False,add_images='',add_masks='', sC=2,model='unet',lr=1e-5,decay=1e-12,
                 timeStamp='',logFileName='',oLabel='',datagen='',checkWeightFileName='',
                 afold=3, numEpochs=1,bs = 1, img_row=512,img_col=512,channel=1,nb_classes=2,
                 classifier = 'softmax',optimizer ='', dtype='float32',dtypeL='uint8',
                 wType='slice', loss='categorical_crossentropy',metric='accuracy'):

        self.root_folder=root_folder
        self.code_folder=code_folder
        self.image_folder=image_folder
        self.mask_folder=mask_folder
        self.save_folder=save_folder
        self.add_images=add_images
        self.add_masks=add_masks
        self.training_folder=training_folder
        self.test_folder=test_folder
        self.pred_folder=pred_folder        
        self.logFileName = logFileName
        self.testingMode=testingMode
        self.testLabelFlag=testLabelFlag
        self.testMetricFlag=testMetricFlag
        self.dataAugmentation=dataAugmentation
        self.savePredMask=savePredMask
        
        self.afold=afold
        self.numEpochs=numEpochs
        self.optimizer = optimizer
        self.oLabel=oLabel
        self.checkWeightFileName = checkWeightFileName
        self.datagen=datagen
        self.dtype=dtype
        self.dtypeL=dtypeL
        self.wType=wType
        self.img_row=img_row
        self.img_col=img_col
        self.channel=channel    
        self.classifier=classifier
        self.bs=bs
        self.sC=sC
        self.nb_classes=nb_classes
        self.includeNC=includeNC
        self.model=model
        self.lr=lr
        self.decay=decay
        self.loss=loss
        self.metric=metric
        
    def __str__(self):
        string = 'Model parameters:\n'
        string += '  root folder: ' + str(self.root_folder) + '\n'
        string += '  code folder: ' + str(self.code_folder) + '\n'
        string += '  image_folder: ' + str(self.image_folder) + '\n'
        string += '  mask folder: ' + str(self.mask_folder) + '\n'
        string += '  save folder: ' + str(self.save_folder) + '\n'
        string += '  training folder: ' + str(self.training_folder) + '\n'
        string += '  prediction folder: ' + str(self.pred_folder) + '\n'
        string += '  log file name: ' + str(self.logFileName) + '\n'
        string += '  testing mode: ' + str(self.testingMode) + '\n'
        string += '  testLabelFlag: ' + str(self.testLabelFlag) + '\n'
        string += '  testMetricFlag: ' + str(self.testMetricFlag) + '\n'
        string += '  dataAugmentation: ' + str(self.dataAugmentation) + '\n'
        string += '  savePredMask: ' + str(self.savePredMask) + '\n'
        
        string += '  augmentation fold: ' + str(self.afold) + '\n'
        string += '  number of epochs: ' + str(self.numEpochs) + '\n'
        string += '  output label: ' + str(self.oLabel) + '\n'
        string += '  checkWeightFileName: ' + str(self.checkWeightFileName) + '\n'
        string += '  data augmentation parameters:'+'\n' + str(self.datagen) + '\n'
        string += '  dtype: ' + str(self.dtype) + '\n'
        string += '  dtypeL: ' + str(self.dtypeL) + '\n'
        string += '  wType: ' + str(self.wType) + '\n'
        string += '  input shape: ' + str([self.img_row,self.img_col,self.channel]) + '\n'   
        string += '  classifier: ' + str(self.classifier) + '\n'
        string += '  batch size: ' + str(self.bs) + '\n'
        string += '  model: ' + str(self.model) + '\n'
        return string

    def arrangeDataPath(self,data_folder,image_folder, mask_folder):
        train_images_path = os.path.join(self.root_folder,data_folder+'/'+image_folder)
        train_labels_path = os.path.join(self.root_folder,data_folder+'/'+mask_folder)
        return train_images_path, train_labels_path

    def arrangeTestPath(self,data_folder,image_folder):
        train_images_path = os.path.join(self.root_folder,data_folder+'/'+image_folder)
        return train_images_path
      
    def loadTrainingData(self,images_path,labels_path, each):
        images = nb.load(os.path.join(images_path,each)).get_data()
        labels = nb.load(os.path.join(labels_path,each)).get_data()
        affine = nb.load(os.path.join(images_path,each)).get_affine()        
        
        if self.dataAugmentation:
            images,labels = self.datagen.generate(images,labels,self.afold,affine)
        [self.img_rows,self.img_cols,self.numImgs] = images.shape
        
        images = images.transpose(2,0,1).reshape(self.numImgs, self.img_rows,self.img_cols,1).astype(self.dtype)
        labels = labels.transpose(2,0,1).reshape(self.numImgs, self.img_rows,self.img_cols,1).astype(self.dtypeL)
        return images,labels, affine

    def loadPredData(self,images_path, each):
        images = nb.load(os.path.join(images_path,each)).get_data()
        affine = nb.load(os.path.join(images_path,each)).get_affine()
        [self.img_rows,self.img_cols,self.numImgs] = images.shape
        images = images.transpose(2,0,1).reshape(self.numImgs, self.img_rows,self.img_cols,1).astype(self.dtype)
        return images, affine

    def load3DtrainingData(self,images_path,labels_path, each):
        images = nb.load(os.path.join(images_path,each)).get_data().astype('float32')
        labels = nb.load(os.path.join(labels_path,each)).get_data().astype('uint8')
        affine = nb.load(os.path.join(images_path,each)).get_affine()
        ind = np.where(labels>0)
        labels[ind]=1

        if len(images.shape)>3:
          images=images[:,:,:,0]

        if self.dataAugmentation:
            images,labels = self.datagen.generate(images,labels,self.afold)
        [self.img_rows,self.img_cols,self.numImgs] = images.shape
        return images,labels, affine
    
    def loadTestData(self,images_path,labels_path, each):
        images = nb.load(os.path.join(images_path,each)).get_data()
        affine = nb.load(os.path.join(images_path,each)).get_affine()
        [self.img_rows,self.img_cols,self.numImgs] = images.shape
        images = images.transpose(2,0,1).reshape(self.numImgs, self.img_rows,self.img_cols,1).astype(self.dtype)
        if self.testLabelFlag:
            labels = nb.load(os.path.join(labels_path,each)).get_data()
            labels = labels.transpose(2,0,1).reshape(self.numImgs, self.img_rows,self.img_cols,1).astype(self.dtypeL)
        else:
            labels = []
        return images,labels, affine
    
    def dice(self,trueL,predL):
        smooth = 1
        trueLF = K.flatten(trueL)
        predLF = K.flatten(predL)
        intersection = K.sum(trueLF * predLF)
        dc = K.eval((2.0 * intersection + smooth) / (K.sum(trueLF) + K.sum(predLF) + smooth))
        print('dice index: ', dc)
        return dc

    def dice_coef(self,y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
      
    def dice_loss(self,trueL,predL):
        return -self.dice_coef(trueL,predL)

    def createModel(self):
        lr=self.lr
        decay=self.decay
        if self.optimizer == 'adam':
            optimizer = Adam(lr = lr, decay=decay)
            self.opt=optimizer
        elif self.optimizer == 'SGD':
            optimizer = SGD(lr = lr, decay=decay)
            self.opt=optimizer
        
        base_model = Unet((self.img_row,self.img_col,self.channel),nb_classes=self.nb_classes)
        act1 = Activation(self.classifier)(base_model.output)
        new_output = Reshape((self.img_row*self.img_col,1,self.nb_classes))(act1)
        top_model = Model(base_model.input,new_output)

        if self.loss=='dice_loss':
          top_model.compile(optimizer=optimizer, loss = self.dice_loss, metrics = [self.dice_coef],sample_weight_mode='temporal')
        else:
          top_model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'],sample_weight_mode='temporal')

        return top_model

    def createModel3D(self,size):
        if self.optimizer is 'adam':
            optimizer = Adam(lr = self.lr, decay=self.decay)
        elif self.optimizer is 'adam':
            optimizer = SGD(lr = self.lr, decay=self.decay)
            
        base_model = Unet3D((size[0],size[1],size[2],self.channel))
        act1 = Activation(self.classifier)(base_model.output)
        new_output = Reshape((size[0]*size[1]*size[2],1,self.nb_classes))(act1)
        top_model = Model(base_model.input,new_output)
        top_model.compile(optimizer=optimizer, loss = self.loss, metrics = ['accuracy'],sample_weight_mode='temporal')
        return top_model
        
    def volumeBasedWeighting(self, trainLabels):
        ind2 = np.where(trainLabels==1)
        ind1 = np.where(trainLabels==0)
        ind3 = np.where(trainLabels==2)
        ind4 = np.where(trainLabels==3)

        wImg = np.zeros(trainLabels.shape, dtype=self.dtypeL)
        if self.nb_classes == 2:
            w1 = ((len(ind1[0]))*1.0/(self.numImgs*self.img_rows*self.img_cols))+np.finfo('float').eps
            w2 = ((len(ind2[0]))*1.0/(self.numImgs*self.img_rows*self.img_cols))+np.finfo('float').eps
    #            w1 = np.median([w1,w2])/w1
    #            w2 = np.median([w1,w2])/w2
            w1 = np.max([w1,w2])/(w1+np.finfo('float').eps)
            w2 = np.max([w1,w2])/(w2+np.finfo('float').eps)
    
            if len(ind1)!=0:
                wImg[ind1] = w1 
            if len(ind2)!=0:
                wImg[ind2] = w2 
        elif self.nb_classes == 3:
            w1 = ((len(ind1[0]))*1.0/(self.numImgs*self.img_rows*self.img_cols))+np.finfo('float').eps
            w2 = ((len(ind2[0]))*1.0/(self.numImgs*self.img_rows*self.img_cols))+np.finfo('float').eps
            w3 = ((len(ind3[0]))*1.0/(self.numImgs*self.img_rows*self.img_cols))+np.finfo('float').eps

            w1 = np.max([w1,w2,w3])/(w1+np.finfo('float').eps)
            w2 = np.max([w1,w2,w3])/(w2+np.finfo('float').eps)
            w3 = np.max([w1,w2,w3])/(w3+np.finfo('float').eps)
            
            if len(ind1)!=0:
                wImg[ind1] = w1 
            if len(ind2)!=0:
                wImg[ind2] = w2     
            if len(ind3)!=0:
                wImg[ind3] = w3  
        elif self.nb_classes==4:
            w1 = ((len(ind1[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
            w2 = ((len(ind2[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
            w3 = ((len(ind3[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
            w4 = ((len(ind4[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
            
            w1 = np.max([w1,w2,w3,w4])/(w1+np.finfo('float').eps)
            w2 = np.max([w1,w2,w3,w4])/(w2+np.finfo('float').eps)
            w3 = np.max([w1,w2,w3,w4])/(w3+np.finfo('float').eps)
            w4 = np.max([w1,w2,w3,w4])/(w4+np.finfo('float').eps)  
            
            if len(ind1)!=0:
                wImg[ind1] = w1 
            if len(ind2)!=0:
                wImg[ind2] = w2    
            if len(ind3)!=0:
                wImg[ind3] = w3 
            if len(ind4)!=0:
                wImg[ind4] = w4            
            
        return wImg
    
    def sliceBasedWeighting(self,trainLabels):
        wImg = np.zeros(trainLabels.shape, dtype=self.dtype)
        for i in range(wImg.shape[0]):
            if self.nb_classes==2:
                ind2 = np.where(trainLabels[i,:]>0)
                ind1 = np.where(trainLabels[i,:]==0)
                w1 = ((len(ind1[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w2 = ((len(ind2[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                
                w1 = np.max([w1,w2])/w1
                w2 = np.max([w1,w2])/w2
                
                if len(ind1)!=0:
                    wImg[i,ind1] = w1 
    
                if len(ind2)!=0:
                    wImg[i,ind2] = w2 
            elif self.nb_classes==3:
                ind3 = np.where(trainLabels[i,:]==2)
                ind2 = np.where(trainLabels[i,:]==1)
                ind1 = np.where(trainLabels[i,:]==0)
                w1 = ((len(ind1[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w2 = ((len(ind2[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w3 = ((len(ind3[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                
                w1 = np.max([w1,w2,w3])/(w1+np.finfo('float').eps)
                w2 = np.max([w1,w2,w3])/(w2+np.finfo('float').eps)
                w3 = np.max([w1,w2,w3])/(w3+np.finfo('float').eps)
                           
                if len(ind1)!=0:
                    wImg[i,ind1] = w1 
                if len(ind2)!=0:
                    wImg[i,ind2] = w2    
                if len(ind3)!=0:
                    wImg[i,ind3] = w3  
            elif self.nb_classes==4:
                ind4 = np.where(trainLabels[i,:]==3)
                ind3 = np.where(trainLabels[i,:]==2)
                ind2 = np.where(trainLabels[i,:]==1)
                ind1 = np.where(trainLabels[i,:]==0)
                w1 = ((len(ind1[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w2 = ((len(ind2[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w3 = ((len(ind3[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w4 = ((len(ind4[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                
                w1 = np.max([w1,w2,w3,w4])/(w1+np.finfo('float').eps)
                w2 = np.max([w1,w2,w3,w4])/(w2+np.finfo('float').eps)
                w3 = np.max([w1,w2,w3,w4])/(w3+np.finfo('float').eps)
                w4 = np.max([w1,w2,w3,w4])/(w4+np.finfo('float').eps)          
                if len(ind1)!=0:
                    wImg[i,ind1] = w1 
                if len(ind2)!=0:
                    wImg[i,ind2] = w2    
                if len(ind3)!=0:
                    wImg[i,ind3] = w3 
                if len(ind4)!=0:
                    wImg[i,ind4] = w4
                        
        return wImg

    def sliceBasedWeighting3D(self,trainLabels):
        wImg = np.zeros(trainLabels.shape, dtype=self.dtype)
        for i in range(wImg.shape[0]):
            if self.nb_classes==2:
                ind2 = np.where(trainLabels[i,:]>0)
                ind1 = np.where(trainLabels[i,:]==0)
                w1 = ((len(ind1[0]))*1.0/(trainLabels.shape[1])+np.finfo('float').eps)
                w2 = ((len(ind2[0]))*1.0/(trainLabels.shape[1])+np.finfo('float').eps)
                
                w1 = np.max([w1,w2])/w1
                w2 = np.max([w1,w2])/w2
                
                if len(ind1)!=0:
                    wImg[i,ind1] = w1 
    
                if len(ind2)!=0:
                    wImg[i,ind2] = w2 
                        
            elif self.nb_classes==3:
                ind3 = np.where(trainLabels[i,:]==2)
                ind2 = np.where(trainLabels[i,:]==1)
                ind1 = np.where(trainLabels[i,:]==0)
                w1 = ((len(ind1[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w2 = ((len(ind2[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w3 = ((len(ind3[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                
                w1 = np.max([w1,w2,w3])/(w1+np.finfo('float').eps)
                w2 = np.max([w1,w2,w3])/(w2+np.finfo('float').eps)
                w3 = np.max([w1,w2,w3])/(w3+np.finfo('float').eps)
                           
                if len(ind1)!=0:
                    wImg[i,ind1] = w1 
                if len(ind2)!=0:
                    wImg[i,ind2] = w2    
                if len(ind3)!=0:
                    wImg[i,ind3] = w3  
            elif self.nb_classes==4:
                ind4 = np.where(trainLabels[i,:]==3)
                ind3 = np.where(trainLabels[i,:]==2)
                ind2 = np.where(trainLabels[i,:]==1)
                ind1 = np.where(trainLabels[i,:]==0)
                w1 = ((len(ind1[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w2 = ((len(ind2[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w3 = ((len(ind3[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                w4 = ((len(ind4[0]))*1.0/(self.img_rows*self.img_cols)+np.finfo('float').eps)
                
                w1 = np.max([w1,w2,w3,w4])/(w1+np.finfo('float').eps)
                w2 = np.max([w1,w2,w3,w4])/(w2+np.finfo('float').eps)
                w3 = np.max([w1,w2,w3,w4])/(w3+np.finfo('float').eps)
                w4 = np.max([w1,w2,w3,w4])/(w4+np.finfo('float').eps)          
                if len(ind1)!=0:
                    wImg[i,ind1] = w1 
                if len(ind2)!=0:
                    wImg[i,ind2] = w2    
                if len(ind3)!=0:
                    wImg[i,ind3] = w3 
                if len(ind4)!=0:
                    wImg[i,ind4] = w4
                        
        return wImg
    
    def train(self):
            
        model = self.createModel()

        print('-'*30)
        print('Loading training data...')
        print('-'*30)
        
        train_images_path, train_labels_path = self.arrangeDataPath(self.root_folder, self.image_folder,self.mask_folder)
        hist={};hist['acc']=[];hist['loss']=[]
        for epochs in range(self.numEpochs):
            print('epochs: ', epochs)
            acc=0;loss=0
            for each in os.listdir(train_images_path):
                print('case: ', each)
                trainImages, trainLabels,affine = self.loadTrainingData(train_images_path,train_labels_path, each)
                print('training image shape:',trainImages.shape)
                trainLabels=trainLabels.reshape((self.numImgs,self.img_rows*self.img_cols))
                if self.wType=='slice':
                    wImg = self.sliceBasedWeighting(trainLabels)
                else:
                    wImg = self.volumeBasedWeighting(trainLabels)
                    
                trainLabels = np_utils.to_categorical(trainLabels, self.nb_classes)
                trainLabels = trainLabels.reshape((self.numImgs,self.img_rows*self.img_cols,1,self.nb_classes))
                trainLabels = trainLabels.astype(self.dtypeL)

                print('-'*30)
                print('Training model...')
                print('-'*30)
        
                history=model.fit(trainImages, trainLabels, batch_size=self.bs, epochs=1, verbose=1,sample_weight=wImg)
                
                acc = acc+history.history['acc'][0]
                loss=loss+history.history['loss'][0]
            if ((epochs>0) and ((epochs+1)%25)==0):
                model.save_weights(os.path.join(self.save_folder,str(epochs+1)+'_'+self.checkWeightFileName))                
            hist['acc'].append(acc/len(os.listdir(train_images_path)))
            hist['loss'].append(loss/len(os.listdir(train_images_path)))
        np.save(self.save_folder+'history.npy',hist)
        return

      
    def train3D(self):
            
        model = self.createModel3D([128,128,48])

        print('-'*30)
        print('Loading training data...')
        print('-'*30)
        
        train_images_path, train_labels_path = self.arrangeDataPath(self.root_folder, self.image_folder,self.mask_folder)
        hist={};hist['acc']=[];hist['loss']=[]
        for epochs in range(self.numEpochs):
            print('epochs: ', epochs)
            acc=0;loss=0
            for each in os.listdir(train_images_path):
                print('case: ', each)
                trainImages, trainLabels,affine = self.load3DtrainingData(train_images_path,train_labels_path, each)
                trainImages = interp3D(trainImages,[0.25,0.25,1],cval=-1024)
                trainLabels = interp3D(trainLabels,[0.25,0.25,1],cval=0)
                trainImages,trainLabels = arrange3Ddata(trainImages,trainLabels,48,self.dtype)

                [numImgs,img_rows,img_cols,img_dep,ch] = trainImages.shape
                print('training image shape:',trainImages.shape)
                trainLabels=trainLabels.reshape((numImgs,img_rows*img_cols*img_dep))
                if self.wType=='slice':
                    wImg = self.sliceBasedWeighting3D(trainLabels)
                else:
#                    wImg = self.volumeBasedWeighting(trainLabels)
                    wImg=np.ones(trainLabels.shape)
                    
                trainLabels = np_utils.to_categorical(trainLabels, self.nb_classes)
                trainLabels = trainLabels.reshape((numImgs,img_rows*img_cols*img_dep,1,self.nb_classes))
                trainLabels = trainLabels.astype(self.dtype)

                print('-'*30)
                print('Training model...')
                print('-'*30)
        
                history=model.fit(trainImages, trainLabels, batch_size=self.bs, epochs=1, verbose=1,sample_weight=wImg)
                acc = acc+history.history['acc'][0]
                loss=loss+history.history['loss'][0]
            if ((epochs>0) and ((epochs+1)%25)==0):
                model.save_weights(os.path.join(self.save_folder,str(epochs+1)+'_'+self.checkWeightFileName))                
#            model.save_weights(os.path.join(self.save_folder,str(epochs+1)+'_'+self.checkWeightFileName))                

            hist['acc'].append(acc/len(os.listdir(train_images_path)))
            hist['loss'].append(loss/len(os.listdir(train_images_path)))
        np.save(self.save_folder+'history.npy',hist)
        return 
    

    
    def saveTestMetrics(self,saveFolder,testLabels,predImage,each):     
        testDataMetrics={};DIL=[];caseList=[];accL=[];recallL=[];roc_aucL = [];cmL=[];precisionL=[];f1_scoreL=[]
#        ind1 = np.where(testLabels!=2)
#        ind2 = np.where(testLabels==2)
#        testLabels[ind1]=0
#        testLabels[ind2]=1
        y_trueL = testLabels.ravel()
        y_predL = predImage.ravel()
        acc = metrics.accuracy_score(y_trueL, y_predL)
        try:
            recall = metrics.recall_score(y_trueL,y_predL)
            roc_auc = metrics.roc_auc_score(y_trueL,y_predL)
            cm = metrics.confusion_matrix(y_trueL,y_predL)
            f1_score = metrics.f1_score(y_trueL,y_predL)
        except:
            recall=''
            roc_auc=''
            cm=''
            f1_score=''
        precision = metrics.precision_score(y_trueL,y_predL)
        DI=self. dice(y_trueL.astype(self.dtype), y_predL)
        
        caseList.append(each);accL.append(acc)
        recallL.append(recall);roc_aucL.append(roc_auc) 
        cmL.append(cm);precisionL.append(precision) 
        f1_scoreL.append(f1_score);DIL.append(DI) 
        testDataMetrics['caseList'] =  caseList
        testDataMetrics['acc'] = accL       
        testDataMetrics['recall'] = recallL 
        testDataMetrics['roc_auc'] = roc_aucL
        testDataMetrics['cm'] = cmL 
        testDataMetrics['precision'] = precisionL
        testDataMetrics['f1_score'] = f1_scoreL
        testDataMetrics['DI'] = DIL         
        if  not os.path.lexists(saveFolder):
            os.mkdir(saveFolder)               
        np.save(saveFolder+'/'+each, testDataMetrics)      
#        np.save(os.path.join(self.save_folder,self.pred_folder+'/'+each[0:-7]), testDataMetrics)      
        return

    def computeTestMetrics(self, testLabels,predImage):
        self.dice(testLabels.astype(self.dtype), predImage)
        return
      
    def Predict(self, weights): 
        test_images_path, test_labels_path = self.arrangeDataPath(self.root_folder,self.image_folder,self.mask_folder)

        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)
        
        model = self.createModel()
        model.load_weights(weights)
        print(test_images_path)
        for each in os.listdir(test_images_path):
            print('case: ', each)
            testImages, testLabels, affine = self.loadTestData(test_images_path,test_labels_path, each)  
            predImage = model.predict(testImages, batch_size=8, verbose=1)       
            print('-'*30)
            print('Predicting masks on test data...')
            print('-'*30) 
            #for comuting metrics of hyper dense class
            predImage = predImage.reshape((self.numImgs,self.img_rows,self.img_cols,self.nb_classes))[:,:,:,self.sC-1:self.sC]
            predImage = (predImage>0.5).astype(self.dtype)
            print('test labels shape: ', predImage.shape)
#            plt.imshow(predImage[15,:,:,2])
#            plt.pause(100)
#            plt.show()
            saveFolder = os.path.join(self.save_folder)
            if self.testLabelFlag:
                self.computeTestMetrics(testLabels,predImage) 
            if self.testMetricFlag: 
                self.saveTestMetrics(saveFolder,testLabels,predImage,each)
            if self.savePredMask:
                predImage = nb.Nifti1Image(predImage[:,:,:,0].transpose(1,2,0), affine)
                nb.save(predImage, saveFolder+'/'+each)
        return


    def Predict3D(self,weights): 
        test_images_path, test_labels_path = self.arrangeDataPath(self.root_folder, self.image_folder,self.mask_folder)

        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)
        
        model = self.createModel3D([128,128,48])
        model.load_weights(os.path.join(self.save_folder,weights))

        from datetime import datetime
        startTime = datetime.now()
        for each in os.listdir(test_images_path):
            print('case: ', each)
            startTime = datetime.now()
            testImages, otestLabels,affine = self.load3DtrainingData(test_images_path,test_labels_path, each)
            oNumImgs=testImages.shape[2]
            testImages = interp3D(testImages,[0.25,0.25,1],cval=-1024)
            testLabels = interp3D(otestLabels,[0.25,0.25,1],cval=0)
            testImages,testLabels = arrange3Ddata(testImages,testLabels,48,self.dtype)
            [numImgs,img_rows,img_cols,img_dep,ch] = testImages.shape
            print('training image shape:',testImages.shape)
            testLabels=testLabels.reshape((numImgs,img_rows*img_cols*img_dep))
            
            testLabels = np_utils.to_categorical(testLabels, self.nb_classes)
            testLabels = testLabels.reshape((numImgs,img_rows*img_cols*img_dep,1,self.nb_classes))
            testLabels = testLabels.astype(self.dtype)

            predImage = model.predict(testImages, batch_size=1, verbose=1)       
            print('-'*30)
            print('Predicting masks on test data...')
            print('-'*30) 
            #for comuting metrics of hyper dense class
            predImage = predImage.reshape((numImgs,img_rows,img_cols,img_dep,self.nb_classes))[:,:,:,:,self.sC-1:self.sC]
            print(predImage.shape)
            predImage = predImage[0,:,:,:,0]
            predImage = interp3D(predImage,[4,4,1],cval=0)
            predImage = (predImage>0.5).astype(self.dtype)
            print('test labels shape: ', predImage.shape)
            print(datetime.now() - startTime)
            
            saveFolder = os.path.join(self.save_folder,self.pred_folder)
            testLabels=testLabels.reshape((numImgs,img_rows,img_cols,img_dep,self.nb_classes))[0,:,:,:,self.sC-1:self.sC]
            if self.testLabelFlag:
                self.computeTestMetrics(otestLabels,predImage) 
            if self.testMetricFlag: 
                self.saveTestMetrics(saveFolder,otestLabels,predImage,each)
            if self.savePredMask:
                predImage = predImage.astype('uint8')
                predImage = nb.Nifti1Image(predImage.reshape(512,512,img_dep), affine)
                nb.save(predImage, saveFolder+'/'+each)
        return
    
