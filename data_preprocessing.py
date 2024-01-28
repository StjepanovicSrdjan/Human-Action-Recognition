import cv2 as cv
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib
import pandas as pd
import os
import mediapipe as mp
from random import randint

nPoints = 18
protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho',
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip',
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

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



def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    contours, _ = cv.findContours(mapMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs(output, frameWidth, frameHeight, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    for k in range(len(mapIdx)):
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv.resize(pafA, (frameWidth, frameHeight))
        pafB = cv.resize(pafB, (frameWidth, frameHeight))

        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            valid_pairs.append(valid_pair)
        else: 
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


def detect(img_path, net):
    image1 = cv.imread(img_path)
    image1 = cv.resize(image1, (240, 240))
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    inHeight = frameHeight
    inWidth = frameWidth

    inpBlob = cv.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    i = 0
    probMap = output[0, i, :, :]
    probMap = cv.resize(probMap, (frameWidth, frameHeight))
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.05

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)
    frameClone = image1.copy()
    valid_pairs, invalid_pairs = getValidPairs(output, frameWidth, frameHeight, detected_keypoints)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)
    return keypoints_list, personwiseKeypoints, frameClone


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

def predict_openpose(key_list, personKeyPoints, model, model_type):
    row = []
    if len(personKeyPoints) == 0:
        return None
    index_of_min_list = min(range(len(personKeyPoints)), key=lambda i: list(personKeyPoints[i]).count(-1))
    for i in range(18):
        index = personKeyPoints[index_of_min_list][i]
        if -1 == index:
            row = row + [0, 0]
            continue
        point = np.int32(key_list[index.astype(int)])
        row += [point[0], point[1]]
    

    if model_type != 'nn':
        row = np.array(row).reshape(1, -1)
        prediction = model.predict(row)
        return classes[prediction[0]]
    else:
        row = np.array(row).reshape(- 1, 18, 2)
        prediction = model.predict(row)
        return(classes[np.argmax(prediction)])

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


# preprocess_mediapipe("/content/drive/MyDrive/ColabNotebookss/OpenPose/HumanActionRecognition/train")
# preprocess_dataset("/content/drive/MyDrive/ColabNotebookss/OpenPose/HumanActionRecognition/train")


def draw_openpose(path, model, model_type):
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    image = cv.imread(path)


    image = cv.resize(image, (240, 240))
    frameWidth = image.shape[1]
    frameHeight = image.shape[0]
    inHeight = frameHeight
    inWidth = frameWidth

    inpBlob = cv.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv.resize(probMap, (image.shape[1], image.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)


    frameClone = image.copy()
    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv.LINE_AA)
    # cv.imshow("Keypoints",frameClone)

    valid_pairs, invalid_pairs = getValidPairs(output, frameWidth, frameHeight, detected_keypoints)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv.LINE_AA)

    frameClone = cv.resize(frameClone, (500, 500))
    class_name = predict_openpose(keypoints_list, personwiseKeypoints, model, model_type)
    cv.rectangle(frameClone, (0, 0), (150, 40), (0, 0, 0), -1)
    cv.putText(frameClone, class_name, (2, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv.imshow("Detected Pose" , frameClone)
    cv.waitKey(0)




