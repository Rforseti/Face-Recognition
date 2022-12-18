# Работает только коллаб...


# Библиотеки
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import seaborn as sns; sns.set()

# Грузим фотки с диска и формируем выборку

import glob
from google.colab import drive
drive.mount('/gdrive', force_remount=True)

# Делаем фотки размера 64х64

from tqdm import tqdm

dirname = "/gdrive/My Drive/Photo"
labels = ['class1', 'class2', 'class3', 'class4']
x, y = [], []
Crop = True
for label in labels:
    subdir = os.path.join(dirname, label)
    filelist = os.listdir(subdir)
    for fname in tqdm(filelist):
        img = cv2.imread(os.path.join(subdir, fname), cv2.IMREAD_GRAYSCALE)
        if Crop is True:
            img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            face = face_cascade.detectMultiScale(img)
            for (a, b, w, h) in face:
                cv2.rectangle(img, (a, b), (a+w, b+h), (0, 0, 255), 2)
                face = img[b:b + h, a:a + w]
        else:
           face = img
        try:
            h, w = face.shape
            size = min(h, w)
            h0 = int((h - size) / 2)
            w0 = int((w - size) / 2)
            img = face[h0: h0 + size, w0: w0 + size]
            img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
            x.append(img)
            y.append(label)
        except:
            print(f'no face found for {fname}')
len(x), len(y)

# Классы фотографий нумеруем

labels = {'AA' : 0, 'Adam_dawson' : 1, 'Boris Johnson' : 2, 'Dimas' : 3, 'Yury_M' : 4}
y = [labels[item] for item in y]

# Выведем первые три изображения, чтобы понять, что все ок

fig, axx = plt.subplots(1, 3, figsize=(10, 5))
for i in range(3):
    img = x[i]
    axx[i].imshow(img)

# Делаем из фоток однострочные векторы

data = np.asarray([el.ravel() for el in x])

# Сделаем обучающие и тестовые данные

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(data, y, stratify = y, random_state=42)

# Обучим логистическую регрессию и выведем точность

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(Xtrain, ytrain)

from sklearn.metrics import accuracy_score

pred_lr = model_lr.predict(Xtest)
accuracy_score(ytest, pred_lr)

# Выведем матрицу совпадений

fig, ax = plt.subplots(4, 6, figsize=(9,9))
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(64, 64), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel('class'+str(pred_lr[i]),
                   color='black' if pred_lr[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);

# Аугментация

import random
import albumentations as A
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

def get_aug(image):
    angle = np.arange(-10,11,1)
    angle0 = random.choice(angle)
    shift = 0.01*np.arange(-10,11,1)
    shift0 = random.choice(shift)
    transform = A.ShiftScaleRotate(shift_limit=shift0,rotate_limit=angle0,scale_limit=0,p=0.5)
    augmented_image = transform(image=image)['image']
    transform = A.HorizontalFlip(p=0.5)
    return transform(image=augmented_image)['image']

# Сделаем обучающие и тестовые данные

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, stratify = y, random_state=42)
len(Xtrain), len(Xtest)

# проведем аугментацию

XtrainAug = []
ytrainAug = []
for (a,b) in zip(Xtrain,ytrain):
    for i in range(3):
      XtrainAug.append(get_aug(a))
      ytrainAug.append(b)
XtrainAug = np.asarray([el.ravel() for el in XtrainAug])
Xtest = np.asarray([el.ravel() for el in Xtest])
XtrainAug.shape, Xtest.shape

# Запустим обучение

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(XtrainAug, ytrainAug)

from sklearn.metrics import accuracy_score

pred_lr = model_lr.predict(Xtest)
accuracy_score(ytest, pred_lr)

# Теперь снизим размерность изображений для того, чтобы получить более точные предсказания и лучше обучить модель

from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA #Principal Components Analysis
from sklearn.pipeline import make_pipeline

pca = KernelPCA(n_components=200, kernel='poly', random_state=42)
model_lr = LogisticRegression()
model = make_pipeline(pca, model_lr)
model.fit(XtrainAug, ytrainAug)

from sklearn.metrics import accuracy_score
pred_lr = model.predict(Xtest)
accuracy_score(ytest, pred_lr)

# Выведем матрицу совпадений

fig, ax = plt.subplots(4, 6, figsize=(9,9))
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(64, 64), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel('class'+str(pred_lr[i]),
                   color='black' if pred_lr[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);