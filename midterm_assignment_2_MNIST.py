# 1. 자신의 Name / Faculty / Student Number pandas로 출력하기
#Name / Faculty / Student Number
# import pandas as pd
# data = {"Name" : ['Yoo'],
#           "Faculty" : ['mechanical engineering'],
#            "Student Number" : [15182403]}
# data_pandas = pd.DataFrame(data)
# print(data_pandas)

# ################################################################################

import csv as csv
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
np.set_printoptions(threshold=np.inf)

train_df = pd.read_csv("C:/Users/kistrobot1/PycharmProjects/pythonProject3/training/DS_AI_project/MNIST 분류 연습/train.csv",header=0)
# train_df = pd.read_csv("C:/Users/유재형/PycharmProjects/pythonProject2/MNIST 분류 연습/train.csv",header=0)
train_data = train_df.values
X_train = train_data[0::,1::]
Y_train = train_data[0::,0]
X_train, X_test, Y_train, Y_test = train_test_split(train_data[0::,1::], train_data[0::,0], test_size=0.2, random_state=0)

################################################################################
# 2. size of train data (X_train 와 y_train) 출력하기
print(X_train.shape)
print(Y_train.shape)

# ################################################################################
# 3. size of test data (X_test 와 y_test) 출력하기
print(X_test.shape)
print(Y_test.shape)
# ################################################################################
X_train_scaled = X_train /255.0
X_test_scaled = X_test /255.0
#
# # show an image
# import matplotlib.pyplot as plt
# plt.imshow(np.reshape(X_train[0::,1],(28,28)))
# plt.show()
#
# MLP learning
clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=1, warm_start=False)
clf.fit(X_train_scaled, Y_train)
neural_output = clf.predict(X_test_scaled)
print("sgd")
print(accuracy_score(Y_test, neural_output))

# #save the model to disk
pickle.dump(clf, open('finalized_model.sav', 'wb'))  # pickle 패키지를 사용

output = neural_output
predictions_file = open("neural_output.csv", "w")
open_file_object = csv.writer(predictions_file)

###############################################################################
#학습된 모델 불러와서 X_test, y_test 중에서 10개의 데이터에 대해서 모델값을 예측하고 정확도 산출하기

# X_test_scaled에서 10개의 데이터 선택
X_exam = X_test_scaled[1000:1010]
Y_exam = Y_test[1000:1010]

Y_exam_predict = clf.predict(X_exam)
print(accuracy_score(Y_exam, Y_exam_predict))



