from copyreg import pickle
import mediapipe as mp
import cv2 as cv
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import draw_openpose
import sys


def load_nn_model(isOpenPose=True):
    if isOpenPose:
        model = load_model("models/HAR_model_op")
    else:
        model = load_model("models/HAR_model_mp")
    print(model.summary())
    return model


def load_svm_model(isOpenPose=True):
    if isOpenPose:
        with open("models/svm_op.pickle", "rb") as file:
            model = pickle.load(file)
    else:
        with open("models/svm_mp.pickle", "rb") as file:
            model = pickle.load(file)
    return model


def load_random_forest_model(isOpenPose=True):
    if isOpenPose:
        with open("models/random_forest_op.pickle", "rb") as file:
            model = pickle.load(file)
    else:
        with open("models/random_forest_mp.pickle", "rb") as file:
            model = pickle.load(file)
    return model

def load_classificator(model_type, isOpenPose=True):
    if model_type == 'nn':
        return load_nn_model(isOpenPose)
    elif model_type == 'svm':
        return load_svm_model(isOpenPose)
    else:
        return load_random_forest_model(isOpenPose)

def process_image_openpose(img_path, model, model_type):
    draw_openpose(img_path, model, model_type)


if __name__=='__main__':
    isOpenPose = True
    if sys.argv[1].lower().strip() == 'false':
        isOpenPose = False

    model_type = sys.argv[2].lower().strip()
    print(len(model_type))

    model = load_classificator(model_type, isOpenPose)
    process_image_openpose('data/HAR/HumanActionRecognition/train/Image_37.jpg', model, model_type)
        
