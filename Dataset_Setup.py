import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mping
import random
from Visualize_Image import view_random_image
#Create Data Set########################################################################################################################################
#Created by Kim M Pambid 11/17/2023 Time 6:00pm MNL Time
#Remember the rules!!!###
    #Dont leak the image for each folder
    #1000 Cat images
    #1000 Dog images
    #Train data 70%  ~ 700 Cat and dog images
    #Test data 15% ~ 150 Cat and Dog images
    #Test data is the validate data
    #Get the Labels or cathegories it
    #rescale it
#download the images from then  rescale and follow the rules above 
#Image Source: https://www.kaggle.com/datasets/tongpython/cat-and-dog
######################################################################################################################################################## 
#Get the classname or labels programmatically
path_dir = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Train_Data"
class_names = os.listdir("E:\Programming\Projects\py_tensorflow\Portfolio 01\Train_Data")

#Let's visualize the image
view_random_image(target_dir=path_dir,target_class="cats")
view_random_image(target_dir=path_dir,target_class="dogs")


