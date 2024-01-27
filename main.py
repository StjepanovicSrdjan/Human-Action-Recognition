from copyreg import pickle
import mediapipe as mp
import cv2 as cv
import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import draw_openpose


def load_nn_model(isOpenPose=True):
    if isOpenPose:
        model = load_model("/models/HAR_model_op")
    else:
        model = load_model("/models/HAR_model_mp")
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
        with open("models/random_forest_op.pickle", "rb") as file:
            model = pickle.load(file)
    return model

def process_image_openpose(model, img_path):
    draw_openpose(img_path)


if __name__=='__main__':
    # while True:
    #     print('\n1. OpenPose\n2.MediaPipe')
    #     isOpenPose = input('\nInput the number of the desired option: ')
    #     if isOpenPose.strip() == '1':
    #         isOpenPose = True
    #     else:
    #         isOpenPose = False

    #     print('\n1. Neural network\n2. SVM\n3.Random Forest')
    #     model_type = input('\nInput the number of the desired option: ')

    #     if model_type.strip() == '1':
    #         model = load_nn_model(isOpenPose)
    #     elif model_type.strip() == '2':
    #         model = load_random_forest_model(isOpenPose)
    #     elif model_type.strip() == '3':
    #         model = load_svm_model(isOpenPose)
    #     else:
    #         break

    process_image_openpose(None, 'data/HAR/HumanActionRecognition/train/Image_37.jpg')
        
