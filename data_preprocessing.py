import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from openpose import detect, getKeypoints, getPersonwiseKeypoints, getValidPairs

nPoints = 18
protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

columns = ['label', 'coordinate 0', 'coordinate 1', 'coordinate 2', 'coordinate 3', 'coordinate 4', 'coordinate 5', 'coordinate 6', 'coordinate 7',
          'coordinate 8', 'coordinate 9', 'coordinate 10', 'coordinate 11', 'coordinate 12', 'coordinate 13', 'coordinate 14', 'coordinate 15', 'coordinate 16',
          'coordinate 17', 'coordinate 18', 'coordinate 19', 'coordinate 20', 'coordinate 21', 'coordinate 22', 'coordinate 23', 'coordinate 24', 'coordinate 25',
          'coordinate 26', 'coordinate 27', 'coordinate 28', 'coordinate 29', 'coordinate 30', 'coordinate 31', 'coordinate 32', 'coordinate 33', 'coordinate 34',
          'coordinate 35']
columns_mp = ['label', 'coordinate 0', 'coordinate 1', 'coordinate 2', 'coordinate 3', 'coordinate 4', 'coordinate 5', 'coordinate 6', 'coordinate 7',
          'coordinate 8', 'coordinate 9', 'coordinate 10', 'coordinate 11', 'coordinate 12', 'coordinate 13', 'coordinate 14', 'coordinate 15', 'coordinate 16',
          'coordinate 17', 'coordinate 18', 'coordinate 19', 'coordinate 20', 'coordinate 21', 'coordinate 22', 'coordinate 23', 'coordinate 24', 'coordinate 25',
          'coordinate 26', 'coordinate 27', 'coordinate 28', 'coordinate 29', 'coordinate 30', 'coordinate 31', 'coordinate 32', 'coordinate 33', 'coordinate 34',
          'coordinate 35', 'coordinate 36', 'coordinate 37', 'coordinate 38', 'coordinate 39', 'coordinate 40', 'coordinate 41', 'coordinate 42', 'coordinate 43',
             'coordinate 44', 'coordinate 45', 'coordinate 46', 'coordinate 47', 'coordinate 48', 'coordinate 49', 'coordinate 50', 'coordinate 51', 'coordinate 52',
             'coordinate 53', 'coordinate 54', 'coordinate 55', 'coordinate 56', 'coordinate 57', 'coordinate 58', 'coordinate 59', 'coordinate 60', 'coordinate 61',
             'coordinate 62', 'coordinate 63', 'coordinate 64', 'coordinate 65']

classes = ['CYCLING', 'DRINKING', 'RUNNING', 'SITTING', 'SLEEPING']



def preprocess_dataset(path):
    labels_df = pd.read_csv("/content/drive/MyDrive/ColabNotebookss/OpenPose/Training_set10to12.csv")
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    output_df = pd.DataFrame(columns=columns)
    for filename in labels_df['filename']:
        row = []
        file_path = os.path.join(path, filename)
        label = labels_df[labels_df['filename'] == filename]['label'].values[0]
        row.append(label)
        key_list, personKeyPoints, _ = detect(file_path, net)
        if len(personKeyPoints) == 0:
            continue
        index_of_min_list = min(range(len(personKeyPoints)), key=lambda i: list(personKeyPoints[i]).count(-1))
        for i in range(18):
            index = personKeyPoints[index_of_min_list][i]
            if -1 == index:
                row = row + [0, 0]
                continue
            point = np.int32(key_list[index.astype(int)])
            row += [point[0], point[1]]
        output_df.loc[len(output_df)] = row
    output_df.to_csv('/content/drive/MyDrive/ColabNotebookss/OpenPose/HAR-dataset10to12.csv', index=False)


def preprocess_mediapipe(path):
    labels_df = pd.read_csv("/content/drive/MyDrive/ColabNotebookss/OpenPose/HumanActionRecognition/Training_set.csv")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    output_df = pd.DataFrame(columns=columns_mp)
    for filename in labels_df['filename']:
        row = []
        file_path = os.path.join(path, filename)
        label = labels_df[labels_df['filename'] == filename]['label'].values[0]
        row.append(label)
        image = cv.imread(file_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb = cv.resize(image_rgb, (240, 240))
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for landmark in landmarks:
                height, width, _ = image.shape
                x, y = landmark.x * width, landmark.y * height
                row += [x, y]
            output_df.loc[len(output_df)] = row
        else:
            continue

    output_df.to_csv('dataset_mp_F.csv')


if __name__=='__main__':
    preprocess_mediapipe("/content/drive/MyDrive/ColabNotebookss/OpenPose/HumanActionRecognition/train")
#   preprocess_dataset("/content/drive/MyDrive/ColabNotebookss/OpenPose/HumanActionRecognition/train")
    