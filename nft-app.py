#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:59:22 2021

@author: juju
"""
import pandas as pd
from difflib import SequenceMatcher
import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Dense, Reshape, Activation
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from PIL import Image
import os
from os import listdir
from os.path import join
import streamlit as st
import pickle
from difflib import SequenceMatcher
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy import ndimage, misc
from skimage.transform import resize
from skimage import data

#NFT GENERATOR:
@st.cache(allow_output_mutation=True)
def load_my_model():
    model = load_model('image_autoencoder_2.h5')
    model.make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    return model

def make_prediction(image,model):
    raw_image = load_img(image, target_size=(256, 256))
    image=img_to_array(raw_image)
    image=np.expand_dims(image,axis=0)
    image=image/255.0
    pred=model.predict(image)
    pred=pred*255.0
    pred=np.reshape(pred,(256,256,3))
    pred=array_to_img(pred)
    return pred
    

#the only way I've been able to get the appropriate image format is by using keras' flow_from_directory, pretending it's a training set
#there is absolutely a better way to do this :) for the sake of time, I will leave it as is (for now)
def get_proper_img_format():
    train_datagen2 = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2)
    
    testing_selfies='folder'
    test_imgs = train_datagen2.flow_from_directory(directory=testing_selfies,
         target_size = (256, 256),
         batch_size = 16,
         class_mode = 'input',
         subset = 'training',
         shuffle=True)
    return test_imgs


#NFT RECOMMENDER
    
def get_name_lst_sd_ghxsts():
    #with open('name_list_combined.pkl', 'rb') as f:
    with open('sd_ghxsts_name_lst.pkl','rb') as f:
        name_lst=pickle.load(f)
        return name_lst
    
def get_name_lst_cpfl():
    with open('cp_fl.pkl','rb') as f:
        name_lst=pickle.load(f)
        return name_lst
        
        
        
def get_images(path):
    reference_image_1 = Image.open(path)
    reference_image_1=reference_image_1.resize((256,256))
    reference_image_arr = np.asarray(reference_image_1)
    #print(np.shape(reference_image_arr))
    return reference_image_1,reference_image_arr

def get_array(img_arr):
    array1=img_arr
    return array1

def flatten(arr):
    flat_array_1 = arr.flatten()
    #print(np.shape(flat_array_1))
    return flat_array_1

def get_rh(flat_arr):
    RH = Counter(flat_arr)
    return RH

def get_h(RH):
    H = []
    for i in range(256):
        if i in RH.keys():
            H.append(RH[i])
        else:
            H.append(0)
    return H

def L2Norm(H1,H2):
    distance =0
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    return np.sqrt(distance)


    
    
if __name__=='__main__':
    count=0
    menu=['NFT Recommender 1','NFT Recommender 2','NFT Image Generator', 'Stats']
    choice=st.sidebar.selectbox('Menu',menu)
    if choice=='NFT Image Generator':
        st.subheader('NFT Image Generator')
        st.title('NFT Generator')
        model=load_my_model()
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image.save(r'folder/training/image.jpeg')
            img_dir=get_proper_img_format()
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Finding generated NFT...")
            img_dir=img_dir.filepaths[-1]
            for i in img_dir:
                
                pred = make_prediction(img_dir,model)
            st.image(pred)
    elif choice=='NFT Recommender 1':
        #st.subheader('NFT Recommender')
        st.title('NFT Recommendations')
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            count+=1
            image = Image.open(uploaded_file)
            image.save(r'recommender-photos/image_{}.jpeg'.format(count))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Finding recommendations...")
            input_image='recommender-photos/image_{}.jpeg'.format(count)
            raw_image = load_img(input_image, target_size = (256, 256))
            #plt.imshow(raw_image)
            #plt.show()
            input_image,image_arr=get_images(input_image)
            in_arr=get_array(image_arr)
            in_flat_arr=flatten(in_arr)
            input_rh=get_rh(in_flat_arr)
            input_h=get_h(input_rh)
            
            name_lst=get_name_lst_sd_ghxsts()
            dic={}
            distances=[]
            for i in name_lst:
                path='images/combined-images-final/{}'.format(i)
                img,img_arr=get_images(path)
                arr=get_array(img_arr)
                flat_arr=flatten(arr)
                rh=get_rh(flat_arr)
                h=get_h(rh)
                dist=L2Norm(input_h,h)
                dic.update({i:dist})
                distances.append(dist)
                
            sorted_dic={k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}
            top_matches = {k: sorted_dic[k] for k in list(sorted_dic)[:5]}
            top_img_names=[]
            for key, value in top_matches.items():
                top_img_names.append(key)
                
            top_imgs=[]
            for i in top_img_names:
                img = load_img('images/sd_ghxsts/{}'.format(i), target_size = (256, 256))
                
                top_imgs.append(img)
            for raw_image in top_imgs:
                plt.imshow(raw_image)
                plt.show()
                st.image(raw_image)
    elif choice=='NFT Recommender 2':
        #st.subheader('NFT Recommender')
        st.title('NFT Recommendations')
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            count+=1
            image = Image.open(uploaded_file)
            image.save(r'recommender-photos/image_{}.jpeg'.format(count))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Finding recommendations...")
            input_image='recommender-photos/image_{}.jpeg'.format(count)
            raw_image = load_img(input_image, target_size = (256, 256))
            #plt.imshow(raw_image)
            #plt.show()
            input_image,image_arr=get_images(input_image)
            in_arr=get_array(image_arr)
            in_flat_arr=flatten(in_arr)
            input_rh=get_rh(in_flat_arr)
            input_h=get_h(input_rh)
            
            name_lst=get_name_lst_cpfl()
            dic={}
            distances=[]
            for i in name_lst:
                path='images/combined-images-final/{}'.format(i)
                #path='iamges/cp_fl/{}'.format(i)
                img,img_arr=get_images(path)
                arr=get_array(img_arr)
                flat_arr=flatten(arr)
                rh=get_rh(flat_arr)
                h=get_h(rh)
                dist=L2Norm(input_h,h)
                dic.update({i:dist})
                distances.append(dist)
                
            sorted_dic={k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}
            top_matches = {k: sorted_dic[k] for k in list(sorted_dic)[:5]}
            top_img_names=[]
            for key, value in top_matches.items():
                top_img_names.append(key)
                
            top_imgs=[]
            for i in top_img_names:
                img = load_img('images/cp_fl/{}'.format(i), target_size = (256, 256))
                
                top_imgs.append(img)
            for raw_image in top_imgs:
                plt.imshow(raw_image)
                plt.show()
                st.image(raw_image)
                
    elif choice=='Stats':
        st.title('NFT Twitter Stats')
        img='Dashboard 2.png'
        st.image(img, use_column_width=True)
        img2='Dashboard 1.png'
        st.image(img2,use_column_width=True)