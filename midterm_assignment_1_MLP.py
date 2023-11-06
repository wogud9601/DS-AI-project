#라이브러리 불러오기

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

import tensorflow as tf
import numpy as np


wine = load_wine()
features = wine ['data']
feature_names = wine['feature_names']
print (feature_names)

df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
df['target'] = wine['target']
df.head()


x= df.iloc[:,:13]
x.head()


y= df['target']
y.head()


x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=0)

#데이터 전처리
# sklearn 내에 preprocessing의 StandardScaler를 로드
# StandardScaler는 정규화를 시키는 함수( 데이터의 범위를 평균 0, 표준편차 1의 범위로 바꿔주는)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)


# X_train과 X_test를 StandardScaler를 이용해 정규화
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#다중인공신경망(MLP) 분류 알고리즘을 sklearn의 neural_network에서 로드
# MLP 알고리즘의 히든레이어를 3계층(10,10,10)으로 할당
from sklearn.preprocessing import StandardScaler
mlp = MLPClassifier(hidden_layer_sizes=(10), activation='logistic',solver='sgd', batch_size=32,
                   learning_rate_init=0.1)


# 위에서 분류한 X_train과 y_train을 MLP를 이용해 학습
# mlp로 학습한 내용을 X_test에 대해 예측하여 predictions변수에 저장
mlp.fit(x_train,y_train)
predictions = mlp.predict(x_test)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))


# precision = 정밀도 = TP/(TP+FP) : 예측을 중심으로 생각
# recall = 재현율 TP/(TP+TN) : 실제값을 중심으로 생각
# F1 Score = 2*(정밀도*재현율)/(정밀도+재현율)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))





