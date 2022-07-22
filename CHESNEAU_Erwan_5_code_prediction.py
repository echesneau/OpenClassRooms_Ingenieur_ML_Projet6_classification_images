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
import pickle
import numpy as np
import tensorflow as tf

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
                        help="Neural network model (tensorflow format)",
                        default="final_model/weights_last_dense.27-0.7832.h5")
    parser.add_argument('--labels', '-l',\
                        help="Labels class names",\
                        default='final_model/class_names.pkl')
    parser.add_argument('--verbose', '-v',\
                        help="Verbose mode",\
                        action="store_true")
    args = parser.parse_args()

    #===========================
    # Download model
    #===========================
    MODEL = tf.keras.models.load_model(args.model)
    config = MODEL.get_config()
    _, HEIGHT, WIDTH, DEPTH = config['layers'][0]['config']['batch_input_shape']
    if args.verbose :
        print(f"input shape = ({HEIGHT}, {WIDTH}, {DEPTH})")
    with open(args.labels, 'rb') as f :
        class_names = pickle.load(f)
    if args.verbose :
        print(f"Number of class : {len(class_names)}")
    #===========================
    # Image preprocessing
    #===========================
    if os.path.isfile(args.ifile) :
        img = tf.keras.utils.load_img(args.ifile, target_size=(HEIGHT, WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        if args.verbose :
            print(f"Image shape = {img_array.shape}")
    else :
        print(f"Error : {args.ifile} is not a file")
        sys.exit()
    #===========================
    # Prediction
    #===========================
    predictions = MODEL.predict(img_array)
    if args.verbose :
        print(f"Prediction shape = {predictions.shape}")
    #===========================
    # Print result
    #===========================
    print("="*(10+len("Results")+2))
    print("|"+" "*5+"Results"+" "*5+'|')
    print("="*(10+len("Results")+2))
    print(f"This image most likely belongs to {class_names[np.argmax(predictions[0])]} " + \
          f"with a {100 * np.max(predictions[0]):.2f} percent confidence.")
