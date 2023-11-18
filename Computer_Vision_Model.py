#save model URL: https://www.youtube.com/watch?v=NVY0FucNRU4
import tensorflow as tf
from tensorflow import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation
from Dataset_Setup import *
from Plot_History_Model import plot_curve_history
#Create Model########################################################################################################################################
#Created by Kim M Pambid 11/18/2023 Time 2:00pm MNL Time
     #Model Rules
     #a. Create a model
     #b. Compile a model
     #c. Fit model 
     #d. Evaluate the model
     #c. save the whole model
######################################################################################################################################################    
#file path
test_path = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Train_Data"
train_path = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 01\\Train_Data"

#get the following data's
train_data_augmented = Data_Sets(test_path_dir = test_path , train_path_dir = train_path).train_scale_images()
test_data = Data_Sets(test_path_dir = test_path , train_path_dir = train_path).test_scale_images()

print(train_data_augmented)

#a. Create a model
tf.random.set_seed=42 
model_of_kim_pambid  = Sequential([
    Conv2D(filters=10, kernel_size=3,input_shape=(224,224,3)),
    Activation(activation="relu"),
    MaxPool2D(),
    Conv2D(filters=10, kernel_size=3,activation="relu"),
    Conv2D(filters=10, kernel_size=3,activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(units=2, activation="softmax") #we only classify 2 types cats & dogs 
    ])
#b. Compile a Model
model_of_kim_pambid.compile(loss="categorical_crossentropy",
                            optimizer=tf.keras.optimizers.Adam(),
                            metrics=["accuracy"])
#c. Fit and save the model
kim_pambid_history= model_of_kim_pambid.fit(train_data_augmented,
                                            epochs=5,
                                            steps_per_epoch=len(train_data_augmented),
                                            validation_data=test_data,
                                            validation_steps=len(test_data))

#d. Evaluate the model
plot_curve_history(history=kim_pambid_history)


#e. Save the whole model not the HDFS 
model_of_kim_pambid.save("computer_vision_model")