import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# Iris : classifier (DNN)

iris = pd.read_csv("E:/iris.data", names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

iris.groupby('species').size()

# iris 5-fold cross-validation data 분류
shuffle_iris = sk.utils.shuffle(iris)
# print(shuffle_iris)
test1 = shuffle_iris[0:30]
training1 = shuffle_iris[30:150]
# print(test1)
# print(training1)
test2 = shuffle_iris[30:60]
training2 = pd.concat([shuffle_iris[0:30], shuffle_iris[60:150]])
# print(test2)
# print(training2)
test3 = shuffle_iris[60:90]
training3 = pd.concat([shuffle_iris[0:60], shuffle_iris[90:150]])
# print(test3)
# print(training3)
test4 = shuffle_iris[90:120]
training4 = pd.concat([shuffle_iris[0:90], shuffle_iris[120:150]])
# print(test4)
# print(training4)
test5 = shuffle_iris[120:150]
training5 = shuffle_iris[0:120]
# print(test5)
# print(training5)

# dnn
# 4개의 입력 노드 + 64개의 노드를 가지는 은닉층 2개 + 3개의 출력노드
# softmax
model = Sequential()
model.add(Dense(64, input_shape=(4,), activation='relu'))
model.add((Dense(64, activation='relu')))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# 100 세대를 테스트
def dnn(training_iris, test_iris):
    # iris 5-fold cross-validation data 분류
    x_train = training_iris.iloc[:, 0:4].values
    y_train = training_iris.iloc[:, 4].values
    x_test = test_iris.iloc[:, 0:4].values
    y_test = test_iris.iloc[:, 4].values

    enc = LabelEncoder()
    y1_train = enc.fit_transform(y_train)
    y_train = pd.get_dummies(y1_train).values
    y1_test = enc.fit_transform(y_test)
    y_test = pd.get_dummies(y1_test).values

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

    y_result = model.predict(x_test)
    y_tp = np.argmax(y_test, axis=1)
    y_pre = np.argmax(y_result, axis=1)

    target = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    print('\n------------------------------------------------------------------------------------')
    print(classification_report(y_tp, y_pre, target_names=target))
    print(confusion_matrix(y_tp, y_pre))
    print('------------------------------------------------------------------------------------\n')


dnn(training1, test1)
dnn(training2, test2)
dnn(training3, test3)
dnn(training4, test4)
dnn(training5, test5)
