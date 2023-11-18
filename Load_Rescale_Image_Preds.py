import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#load and rescale data using OOP###################################################
#Created by Kim M Pambid 11/18/2023 Time 3:00pm MNL Time
#create a function to import and image and resize it to be able to used with our model
class Image_Rescale_Preds:
    def __init__(self, sample_file_path, load_model, class_names, img_shape):
       self.filename = sample_file_path
       self.model = load_model
       self.class_names = class_names
       self.img_shape = img_shape
    
    def pred_and_plot(self):
        """
        Import an image located at filename, makes a prediction with model and
        plot the image with the predicted class as the title.        """
        
        #import the target image and preprocess it
        #Read in the image
        model = self.model
        img = tf.io.read_file(self.filename)

        #Decode the read file into a tensor
        img = tf.image.decode_image(img)

        #resize the image
        img = tf.image.resize(img, size=[self.img_shape,self.img_shape])
        #Rescale the image(get all avlues between 0 & 1)
        img=img/255.
        
        #make a prediction
        pred=model.predict(tf.expand_dims(img,axis=0))

        #Add in logic for multi-class
        if len(pred[0])>1:
            pred_class = self.class_names[tf.argmax(pred[0])] #multi-class
        else:
            pred_class = self.class_names[int(tf.round(pred[0]))] #binary

        #plot the image and predicted class
        plt.imshow(img)
        plt.title(f"Prediction:{pred_class}")
        plt.axis(False)
        plt.show()
   
        