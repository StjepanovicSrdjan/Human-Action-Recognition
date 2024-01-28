from copyreg import pickle
import mediapipe as mp
import cv2 as cv
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from openpose import draw_openpose
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


def mediapipe_pose_detection():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv.imshow("Pose Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__=='__main__':
    isOpenPose = True
    if sys.argv[1].lower().strip() == 'false':
        isOpenPose = False
        mediapipe_pose_detection()
    else:
        model_type = sys.argv[2].lower().strip()
        print(len(model_type))

        model = load_classificator(model_type, isOpenPose)
        process_image_openpose('data/HAR/HumanActionRecognition/train/Image_37.jpg', model, model_type)
        
