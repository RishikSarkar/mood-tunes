import os
import pandas as pd
from keras.models import load_model
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

model_path = os.path.join('models', 'model.h5')
model = load_model(model_path)

song_dataset = pd.read_csv(os.path.join('data', 'song_moods.csv'))
song_dataset = song_dataset[['name', 'artist', 'mood']]

#'''
cap = cv2.VideoCapture(0)

time.sleep(3)
print('Collecting test image')
ret, frame = cap.read()
imgname = os.path.join('collectedImages', 'test_image.jpg')
cv2.imwrite(imgname, frame)
cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
#'''

test_img_dir = os.path.join('collectedImages', 'test_image.jpg')
cropped_test_img_dir = os.path.join('collectedImages', 'cropped', 'test_cropped.jpg')

img = cv2.imread(test_img_dir)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

haarscascade_dir = os.path.join('haarscascade', 'haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(haarscascade_dir)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    faces = img[y:y + h, x:x + w]
    cv2.imwrite(cropped_test_img_dir, faces)

if not os.path.exists(cropped_test_img_dir):
    cropped_test_img_dir = test_img_dir

test_image = cv2.imread(cropped_test_img_dir, cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (48, 48))

test_img = np.array(test_image)
test_img = test_img.reshape(1, 48, 48, 1)

def mood_from_label(label):
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    return labels[label]

predict_x = model.predict(test_img)
result = np.argmax(predict_x, axis=1)

mood_label = result[0]
mood = mood_from_label(result[0])

def find_song_mood(label):
    if (label == 0 or label == 1 or label == 2):
        return 'Calm'
    elif (label == 3 or label == 4):
        return 'Happy'
    elif (label == 5):
        return 'Sad'
    elif (label == 6):
        return 'Energetic'

song_mood = song_dataset['mood'] == find_song_mood(mood_label)

temp = song_dataset.where(song_mood)
temp = temp.dropna()
songs = temp.sample(n=5)
songs.reset_index(inplace=True)

print(mood)
print(songs)

if os.path.exists(test_img_dir):
    os.remove(test_img_dir)
if os.path.exists(cropped_test_img_dir):
    os.remove(cropped_test_img_dir)