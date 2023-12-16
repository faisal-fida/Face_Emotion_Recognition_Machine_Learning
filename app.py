from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import time

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

webcam = cv2.VideoCapture(0)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def cv2_helper():
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    for p, q, r, s in faces:
        image = gray[q : q + s, p : p + r]
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        cv2.putText(
            im,
            "% s" % (prediction_label),
            (p - 10, q - 10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            2,
            (0, 0, 255),
        )
    cv2.imshow("Output", im)
    cv2.waitKey(27)


while True:
    try:
        cv2_helper()
    except cv2.error as e:
        print("No face detected because of ", e)
        pass
