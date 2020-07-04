

import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet import preprocess_input as preprocess_input_resnet
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering



def extract_features(img_dir_path, model, preprocess_input):

  extracted_features_list = []
  
  img_list = os.listdir(img_dir_path)

  for img in img_list:

    img = os.path.join(img_dir_path, img)
    image = cv2.imread(img)
    image = cv2.resize(image,(224,224))

    image = preprocess_input(np.expand_dims(image.copy(), axis=0))
    
    feature = model.predict(image)
    feature_np = np.array(feature)
    extracted_features_list.append(feature_np.flatten())

  return np.array(extracted_features_list)

def get_vgg19():

    vgg19 = VGG19(include_top=False)
    preprocess_input = preprocess_input_vgg19

    return (vgg19, preprocess_input)

def get_resnet50():

    resnet50 = ResNet50(include_top=False, pooling='avg') # used avg pooling here 
    preprocess_input = preprocess_input_resnet 

    return (resnet50, preprocess_input)


def pred_KMeans(clusters, features):

    model = KMeans(n_clusters = clusters,random_state=0)
    labels = model.fit_predict(features)

    return labels


def pred_AgglomerativeClustering(clusters, features):

    model = AgglomerativeClustering(n_clusters = clusters)
    labels = model.fit_predict(features)

    return labels











    
    



    
