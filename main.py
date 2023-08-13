import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

def mediapipeDetection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
    
    return image, results

def drawLandmarks(image, results):
    mpDrawing.draw_landmarks(image, results.face_landmarks, mpHolistic.FACEMESH_TESSELATION) #Draw face connections
    mpDrawing.draw_landmarks(image, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS) #Draw pose connections
    mpDrawing.draw_landmarks(image, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS) #Draw left hand connections
    mpDrawing.draw_landmarks(image, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS) #Draw right hand connections

def extractLandmarkCoordinates(points):
    #Put all landmark coordinates into a single array if they exist, otherwise put in an array of zeros
    pose = np.array([[point.x, point.y, point.z, point.visibility] for point in points.pose_landmarks.landmark]).flatten() if points.pose_landmarks else np.zeros(33*4) #33 keypoints, each with 4 values
    leftHand = np.array([[point.x, point.y, point.z] for point in points.left_hand_landmarks.landmark]).flatten() if points.left_hand_landmarks else np.zeros(21*3) #21 keypoints, each with 3 values
    rightHand = np.array([[point.x, point.y, point.z] for point in points.right_hand_landmarks.landmark]).flatten() if points.right_hand_landmarks else np.zeros(21*3) #21 keypoints, each with 3 values
    face = np.array([[point.x, point.y, point.z] for point in points.face_landmarks.landmark]).flatten() if points.face_landmarks else np.zeros(468*3) #468 keypoints, each with 3 values

    return np.concatenate([pose, face, leftHand, rightHand])    

def confidenceVisual(result, actions, inputFrame, colors):
    outputFrame = inputFrame.copy()
    for num, prob in enumerate(result):
        cv2.rectangle(outputFrame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(outputFrame, actions[num], (0,85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return outputFrame

#MediaPipe Solutions
mpHolistic = mp.solutions.holistic #Holistic model
mpDrawing = mp.solutions.drawing_utils #Drawing utilities
capture = cv2.VideoCapture(0)

#Path for exported data
DATA_PATH = os.path.join("MP_Data")

#Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30 #Number of sequences per action
sequence_length = 30 #Number of frames per sequence

#Build the model
model = Sequential()    #Sequential model

#LSTM layer with 64 neurons, return sequences is true because we are stacking LSTM layers. Input shape is 30 sequences of 1662 value
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation="relu")) #LSTM layer with 128 neurons
model.add(LSTM(64, return_sequences=False, activation="relu")) #LSTM layer with 64 neurons
model.add(Dense(64, activation="relu")) #Dense layer with 64 neurons
model.add(Dense(32, activation="relu")) #Dense layer with 32 neurons
model.add(Dense(actions.shape[0], activation="softmax")) #Dense layer with 3 neurons, one for each action

#Compile the model categorical_crossentropy because we have more than 2 classes
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

menuInput = str(input("Enter 1 to record new training data, 2 to train new model, 3 to run model:"))
#Record new training data
if menuInput == "1":
    #Create folders for each action 30 sequences of 30 frames each
    for action in actions:
        #Create subfolders for each sequence
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                #Loop through each frame in the sequence
                for frameNo in range(sequence_length):

                    #hasRead is a boolean regarding whether or not there was a return at all, frame is each frame that is returned
                    hasRead, frame = capture.read()

                    #Detections
                    image, results = mediapipeDetection(frame, holistic)

                    #Draw Landmarks
                    drawLandmarks(image, results)

                    #Apply wait logic
                    if frameNo == 0:
                        cv2.putText(image, "STARTING COLLECTION", (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(image, "Collecting frames for {} Video Number {}".format(action, sequence), (15,125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, "Collecting frames for {} Video Number {}".format(action, sequence), (15,125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                    
                    #Export keypoints
                    keypoints = extractLandmarkCoordinates(results)
                    npyPath = os.path.join(DATA_PATH, action, str(sequence), str(frameNo))
                    np.save(npyPath, keypoints)

                    

                    #Show to screen
                    cv2.imshow("OpenCV Feed", image)

                    # Press "q" to exit
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        
    capture.release()
    cv2.destroyAllWindows()

elif menuInput == "2":
    #Map actions to numbers
    label_map = {label:num for num, label in enumerate(actions)} 
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frameNo in range(sequence_length):
                #Load image
                image = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frameNo)))
                window.append(image)
            sequences.append(window)
            labels.append(label_map[action])
    npSequenceArray = np.array(sequences)
    npSequenceLabels = to_categorical(labels).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(npSequenceArray, npSequenceLabels, test_size=0.05)

    log_dir = os.path.join("Logs")
    tbCallback = TensorBoard(log_dir=log_dir)

    model.fit(x_train, y_train, epochs=100, callbacks=[tbCallback])

    # #Evaluate the model
    model.save("action.h5")

elif menuInput == "3":
    #Load the model
    model.load_weights("action.h5")
    
    #Detection Variables
    sequence = []
    sentence = []
    confidenceThreshold = 0.7

    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while capture.isOpened():
            #hasRead is a boolean regarding whether or not there was a return at all, frame is each frame that is returned
            hasRead, frame = capture.read()

            #Detections
            image, results = mediapipeDetection(frame, holistic)

            #Draw Landmarks
            drawLandmarks(image, results)

            #Prediction Logic
            keypoints = extractLandmarkCoordinates(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                result = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(result)])
        
                #Visualization
                if result[np.argmax(result)] > confidenceThreshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(result)] != sentence[-1]:
                            sentence.append(actions[np.argmax(result)])
                    else:
                        sentence.append(actions[np.argmax(result)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                image = confidenceVisual(result, actions, image, [(245, 117, 16), (117, 245, 16), (16, 117, 245)])



            #Displaying the text
            cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
            cv2.putText(image, " ".join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            

            #Show to screen
            cv2.imshow("OpenCV Feed", image)

            # Press "q" to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    capture.release()
    cv2.destroyAllWindows()

