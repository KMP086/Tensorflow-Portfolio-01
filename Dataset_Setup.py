import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mping
import random
from Visualize_Image import view_random_image
import tensorflow as tf
from tensorflow import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

#Create Data Set########################################################################################################################################
#Created by Kim M Pambid 11/18/2023 Time 1:00pm MNL Time
#Remember the rules!!!###    
    #Dont leak the image for each folder
    #4000 Cat images
    #4000 Dog images
    #Use OOP in programming!!!
    #Train data 70%  ~ 5600 Cat and dog images
    #Test data 30% ~ 2400 Cat and Dog images
    #Test data is the validate data
    #Get the Labels or cathegories it
    #scale it by 225 and augment the train data/images
#download the images from then  rescale and follow the rules above 
#Image Source: https://www.kaggle.com/datasets/tongpython/cat-and-dog
######################################################################################################################################################## 
#Get the classname or labels programmatically
#Let's use OOP in our project 
class Data_Sets: 
    def __init__(self, test_path_dir, train_path_dir):
        self.train_file_path =  train_path_dir  #70% of images
        self.test_file_path =  test_path_dir  #30% of images
         

    def class_names(self):
        return os.listdir(self.train_file_path)

    #Let's visualize the image
    def view_image(self):
        view_random_image(target_dir=self.train_file_path,target_class="cats")
        view_random_image(target_dir=self.train_file_path,target_class="dogs")

    #Scale the data by 225 x 225
    #The images for both Train and test data must be in the same shape.
    #train data should be augmentated(to train the model in different shapes and form)
    def train_scale_images(self):
        tf.random.set_seed=42 #universal number
        #augmenting the data
        train_datagen_augmented = ImageDataGenerator(rescale=1./255.,
                                                     rotation_range=0.2,
                                                     width_shift_range=0.2,
                                                     height_shift_range=0.2,
                                                     zoom_range=0.2,
                                                     horizontal_flip=True)
        #shaping or scaling the the train image
        train_data_augmented = train_datagen_augmented.flow_from_directory(directory = self.train_file_path,
                                                                        batch_size = 32,   
                                                                        target_size=(224,224),
                                                                        class_mode = "categorical",  
                                                                        seed=42)
        return train_data_augmented
    
    def test_scale_images(self):
        tf.random.set_seed=42 #universal number
        test_datagen = ImageDataGenerator(rescale=1./255)
        #shaping or scaling the the test image
        test_data = test_datagen.flow_from_directory(directory = self.test_file_path,
                                                      batch_size = 32,
                                                      target_size=(224,224),
                                                      class_mode = "categorical",
                                                      seed=42)
        
        return test_data

