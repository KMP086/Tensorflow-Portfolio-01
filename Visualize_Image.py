#visualize image
import matplotlib.pyplot as plt   
import matplotlib.image as mping
import random 
import os
def view_random_image(target_dir, target_class):
    #set target folder 
    target_folder =  target_dir + "\\" + target_class
    
    #get random image from path
    random_image = random.sample(os.listdir(target_folder), 1)
    
    #reading the image and plot it with matplotlib
    img = mping.imread(target_folder + "\\" + random_image[0])
    print(f"Image Shape:{img.shape}")
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off");
    plt.show()
 
    
      
    