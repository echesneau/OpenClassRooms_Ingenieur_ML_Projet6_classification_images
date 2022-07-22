#!/bin/python3.6

"""
Code to use the CNN developped to predict the dog breed
"""

#===========================
# Python's Module
#===========================
import sys
import os
import argparse
import numpy as np

from tensorflow import expand_dims
from tensorflow import keras, nn
#from tensorflow.keras import layers
#from tensorflow import data

if __name__ == '__main__' :
    #===========================
    # Arguments
    #===========================
    parser = argparse.ArgumentParser(description='Create clustering model '+\
                                     'for questions of StackOverFlow '+\
                                     '\n Written by E. Chesneau')
    parser.add_argument('--ifile', '-i', \
                        help="Path to the image which we want to use for the prediction")
    parser.add_argument('--model', '-m',\
                        help="Neural network model (tensorflow format)")
    args = parser.parse_args()

    #===========================
    # Download model
    #===========================
    MODEL = ""
    #lire la taille des image
    HEIGHT = 0
    WIDTH = 0

    #===========================
    # Image preprocessing
    #===========================
    if os.path.isfile(args.ifile) :
        #image_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
        img = keras.utils.load_img(args.ifile, target_size=(HEIGHT, WIDTH))
        img_array = keras.utils.img_to_array(img)
        img_array = expand_dims(img_array, 0)
    else :
        print(f"Error : {args.ifile} is not a file")
        sys.exit()
    #===========================
    # Prediction
    #===========================
    predictions = MODEL.predict(img_array)
    score = nn.softmax(predictions[0])
    #===========================
    # Print result
    #===========================
    print(f"This image most likely belongs to {class_names[np.argmax(score)]} " + \
          f"with a {100 * np.max(score):.2f} percent confidence.")
