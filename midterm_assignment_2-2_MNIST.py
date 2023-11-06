
import csv as csv
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)

################################################################################
#직접 쓴 손글씨 예측하기

# 모델 파일 로드
import pickle
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

from PIL import Image
import os

image_folder = "C:/Users/kistrobot1/PycharmProjects/pythonProject3/training/DS_AI_project/MNIST 분류 연습"
# image_folder = "C:/Users/유재형/PycharmProjects/pythonProject2/MNIST 분류 연습"
for i in range(0, 10):
    # 이미지 파일 경로 생성
    image_path = os.path.join(image_folder, f"test{i}.png")
    # 이미지 불러오기
    image = Image.open(image_path)
    # 이미지 크기를 MNIST와 동일하게 조정 (28x28 픽셀)
    image = image.resize((28, 28)).convert("L")
    # 이미지를 배열로 변환
    image_array = ((np.array(image) / 255) - 1) * -1

    mask1 = image_array <= 0.38  # 0.28 이하의 값에 대한 마스크 생성
    image_array[mask1] = 0  # 해당 값들을 0으로 설정
    mask2 = image_array > 0.28  # 0.28 이상의 값에 대한 마스크 생성
    image_array[mask2] = image_array[mask2] *1.4  # 해당 값들에 곱하기 1.4으로 설정
    # print(image_array)
    import matplotlib.pyplot as plt

    plt.show()
    plt.figure(figsize=(8, 8))  # 그림의 크기를 조정합니다
    plt.imshow(image_array, cmap='viridis')  # 행렬을 이미지로 표시합니다

    # 모델에 입력할 수 있도록 이미지 차원을 맞춤
    image_array = image_array.reshape(1, -1)

    # 모델로 예측
    prediction =loaded_model.predict(image_array)
    print(f"Actual:{i}","prediced:", prediction)


