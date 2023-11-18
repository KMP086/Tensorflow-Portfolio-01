import tensorflow as tf
from tensorflow import *
from Load_Rescale_Image_Preds import *
from Dataset_Setup import *
#Sample prediction################################################################################################
#Created by Kim M Pambid 11/18/2023 Time 4:00pm MNL Time
    #get a sample from the google note it should be 225 x 225 in shape (or greater), 
        #(the sample images would not came from the test and train file)  
    #load the created model
    #rescale the sample model
    #Predict the model
##################################################################################################################    
#get the classnames
test_path = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Train_Data"
train_path = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Train_Data"
classnames = Data_Sets(test_path_dir = test_path , train_path_dir = train_path).class_names()

#load the model
model_file_path = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\computer_vision_model" 
print(f"Model path exist?: {os.path.exists(model_file_path)}")
model_of_kim_pambid = tf.keras.models.load_model(model_file_path)      
model_of_kim_pambid.summary()
#image for prediction##############################################################################################
file_sample_path01 = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Cat and Dog sample\\image01.jpg"
file_sample_path02 = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Cat and Dog sample\\image02.jpg"
file_sample_path03 = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Cat and Dog sample\\image03.jpg"
file_sample_path04 = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Cat and Dog sample\\image04.jpg"
###################################################################################################################

Image_Rescale_Preds(sample_file_path=file_sample_path04, load_model= model_of_kim_pambid, class_names=classnames, img_shape=224).pred_and_plot()