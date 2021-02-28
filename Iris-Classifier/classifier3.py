import numpy as np
import pandas as pd
import sklearn as sk

# Iris : classifier (Multi-variate Gaussian distribution)

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

precision, recall = 0, 0


# classifier 분류 결과와, tp, test data 실제 종 수 출력
# precision, recall 값 출력
def posterior(training_iris, test_iris):
    s_cnt, v_cnt, vi_cnt = 0, 0, 0
    s_tp, v_tp, vi_tp = 0, 0, 0

    mean_sx1, mean_sx2, mean_sx3, mean_sx4 = 0, 0, 0, 0
    mean_vx1, mean_vx2, mean_vx3, mean_vx4 = 0, 0, 0, 0
    mean_vix1, mean_vix2, mean_vix3, mean_vix4 = 0, 0, 0, 0
    var_sx1, var_sx2, var_sx3, var_sx4 = 0, 0, 0, 0
    var_sx1x2, var_sx1x3, var_sx1x4, var_sx2x3, var_sx2x4, var_sx3x4 = 0, 0, 0, 0, 0, 0
    var_vx1, var_vx2, var_vx3, var_vx4 = 0, 0, 0, 0
    var_vx1x2, var_vx1x3, var_vx1x4, var_vx2x3, var_vx2x4, var_vx3x4 = 0, 0, 0, 0, 0, 0
    var_vix1, var_vix2, var_vix3, var_vix4 = 0, 0, 0, 0
    var_vix1x2, var_vix1x3, var_vix1x4, var_vix2x3, var_vix2x4, var_vix3x4 = 0, 0, 0, 0, 0, 0

    # 각 종의 training data 수
    ts = len(training_iris[training_iris.species == 'Iris-setosa'])
    tv = len(training_iris[training_iris.species == 'Iris-versicolor'])
    tvi = len(training_iris[training_iris.species == 'Iris-virginica'])
    # 각 종의 test data 수
    s = len(test_iris[test_iris.species == 'Iris-setosa'])
    v = len(test_iris[test_iris.species == 'Iris-versicolor'])
    vi = len(test_iris[test_iris.species == 'Iris-virginica'])

    # 각 종의 x1, x2, x3, x4 평균
    for j in range(120):
        if training_iris.iloc[j, 4] == 'Iris-setosa':
            mean_sx1 += training_iris.iloc[j, 0]
            mean_sx2 += training_iris.iloc[j, 1]
            mean_sx3 += training_iris.iloc[j, 2]
            mean_sx4 += training_iris.iloc[j, 3]
        elif training_iris.iloc[j, 4] == 'Iris-versicolor':
            mean_vx1 += training_iris.iloc[j, 0]
            mean_vx2 += training_iris.iloc[j, 1]
            mean_vx3 += training_iris.iloc[j, 2]
            mean_vx4 += training_iris.iloc[j, 3]
        else:
            mean_vix1 += training_iris.iloc[j, 0]
            mean_vix2 += training_iris.iloc[j, 1]
            mean_vix3 += training_iris.iloc[j, 2]
            mean_vix4 += training_iris.iloc[j, 3]
    mean_sx1, mean_sx2, mean_sx3, mean_sx4 = mean_sx1 / ts, mean_sx2 / ts, mean_sx3 / ts, mean_sx4 / ts
    mean_vx1, mean_vx2, mean_vx3, mean_vx4 = mean_vx1 / tv, mean_vx2 / tv, mean_vx3 / tv, mean_vx4 / tv
    mean_vix1, mean_vix2, mean_vix3, mean_vix4 = mean_vix1 / tvi, mean_vix2 / tvi, mean_vix3 / tvi, mean_vix4 / tvi
    # 각 종의 평균 배열
    mean_s_arr = np.array([mean_sx1, mean_sx2, mean_sx3, mean_sx4])
    mean_v_arr = np.array([mean_vx1, mean_vx2, mean_vx3, mean_vx4])
    mean_vi_arr = np.array([mean_vix1, mean_vix2, mean_vix3, mean_vix4])

    # 각 종의 x1, x2, x3, x4  분산, 공분산
    for j in range(120):
        if training_iris.iloc[j, 4] == 'Iris-setosa':
            var_sx1 += (training_iris.iloc[j, 0] - mean_sx1) ** 2
            var_sx2 += (training_iris.iloc[j, 1] - mean_sx2) ** 2
            var_sx3 += (training_iris.iloc[j, 2] - mean_sx3) ** 2
            var_sx4 += (training_iris.iloc[j, 3] - mean_sx4) ** 2
            var_sx1x2 += (training_iris.iloc[j, 0] - mean_sx1) * (training_iris.iloc[j, 1] - mean_sx2)
            var_sx1x3 += (training_iris.iloc[j, 0] - mean_sx1) * (training_iris.iloc[j, 2] - mean_sx3)
            var_sx1x4 += (training_iris.iloc[j, 0] - mean_sx1) * (training_iris.iloc[j, 3] - mean_sx4)
            var_sx2x3 += (training_iris.iloc[j, 1] - mean_sx2) * (training_iris.iloc[j, 2] - mean_sx3)
            var_sx2x4 += (training_iris.iloc[j, 1] - mean_sx2) * (training_iris.iloc[j, 3] - mean_sx4)
            var_sx3x4 += (training_iris.iloc[j, 2] - mean_sx3) * (training_iris.iloc[j, 3] - mean_sx4)
        elif training_iris.iloc[j, 4] == 'Iris-versicolor':
            var_vx1 += (training_iris.iloc[j, 0] - mean_vx1) ** 2
            var_vx2 += (training_iris.iloc[j, 1] - mean_vx2) ** 2
            var_vx3 += (training_iris.iloc[j, 2] - mean_vx3) ** 2
            var_vx4 += (training_iris.iloc[j, 3] - mean_vx4) ** 2
            var_vx1x2 += (training_iris.iloc[j, 0] - mean_vx1) * (training_iris.iloc[j, 1] - mean_vx2)
            var_vx1x3 += (training_iris.iloc[j, 0] - mean_vx1) * (training_iris.iloc[j, 2] - mean_vx3)
            var_vx1x4 += (training_iris.iloc[j, 0] - mean_vx1) * (training_iris.iloc[j, 3] - mean_vx4)
            var_vx2x3 += (training_iris.iloc[j, 1] - mean_vx2) * (training_iris.iloc[j, 2] - mean_vx3)
            var_vx2x4 += (training_iris.iloc[j, 1] - mean_vx2) * (training_iris.iloc[j, 3] - mean_vx4)
            var_vx3x4 += (training_iris.iloc[j, 2] - mean_vx3) * (training_iris.iloc[j, 3] - mean_vx4)
        else:
            var_vix1 += (training_iris.iloc[j, 0] - mean_vix1) ** 2
            var_vix2 += (training_iris.iloc[j, 1] - mean_vix2) ** 2
            var_vix3 += (training_iris.iloc[j, 2] - mean_vix3) ** 2
            var_vix4 += (training_iris.iloc[j, 3] - mean_vix4) ** 2
            var_vix1x2 += (training_iris.iloc[j, 0] - mean_vix1) * (training_iris.iloc[j, 1] - mean_vix2)
            var_vix1x3 += (training_iris.iloc[j, 0] - mean_vix1) * (training_iris.iloc[j, 2] - mean_vix3)
            var_vix1x4 += (training_iris.iloc[j, 0] - mean_vix1) * (training_iris.iloc[j, 3] - mean_vix4)
            var_vix2x3 += (training_iris.iloc[j, 1] - mean_vix2) * (training_iris.iloc[j, 2] - mean_vix3)
            var_vix2x4 += (training_iris.iloc[j, 1] - mean_vix2) * (training_iris.iloc[j, 3] - mean_vix4)
            var_vix3x4 += (training_iris.iloc[j, 2] - mean_vix3) * (training_iris.iloc[j, 3] - mean_vix4)
    var_sx1, var_sx2, var_sx3, var_sx4 = var_sx1 / s, var_sx2 / s, var_sx3 / s, var_sx4 / s
    var_vx1, var_vx2, var_vx3, var_vx4 = var_vx1 / v, var_vx2 / v, var_vx3 / v, var_vx4 / v
    var_vix1, var_vix2, var_vix3, var_vix4 = var_vix1 / vi, var_vix2 / vi, var_vix3 / vi, var_vix4 / vi
    var_sx1x2, var_sx1x3, var_sx1x4 = var_sx1x2 / s, var_sx1x3 / s, var_sx1x4 / s
    var_sx2x3, var_sx2x4, var_sx3x4 = var_sx2x3 / s, var_sx2x4 / s, var_sx3x4 / s
    var_vx1x2, var_vx1x3, var_vx1x4 = var_vx1x2 / v, var_vx1x3 / v, var_vx1x4 / v
    var_vx2x3, var_vx2x4, var_vx3x4 = var_vx2x3 / v, var_vx2x4 / v, var_vx3x4 / v
    var_vix1x2, var_vix1x3, var_vix1x4 = var_vix1x2 / vi, var_vix1x3 / vi, var_vix1x4 / vi
    var_vix2x3, var_vix2x4, var_vix3x4 = var_vix2x3 / vi, var_vix2x4 / vi, var_vix3x4 / vi
    # 각 종의 분산 배열
    var_s_arr = np.array([[var_sx1, var_sx1x2, var_sx1x3, var_sx1x4],
                          [var_sx1x2, var_sx2, var_sx2x3, var_sx2x4],
                          [var_sx1x3, var_sx2x3, var_sx3, var_sx3x4],
                          [var_sx1x4, var_sx2x4, var_sx3x4, var_sx4]])
    var_v_arr = np.array([[var_vx1, var_vx1x2, var_vx1x3, var_vx1x4],
                          [var_vx1x2, var_vx2, var_vx2x3, var_vx2x4],
                          [var_vx1x3, var_vx2x3, var_vx3, var_vx3x4],
                          [var_vx1x4, var_vx2x4, var_vx3x4, var_vx4]])
    var_vi_arr = np.array([[var_vix1, var_vix1x2, var_vix1x3, var_vix1x4],
                           [var_vix1x2, var_vix2, var_vix2x3, var_vix2x4],
                           [var_vix1x3, var_vix2x3, var_vix3, var_vix3x4],
                           [var_vix1x4, var_vix2x4, var_vix3x4, var_vix4]])

    # test_iris x1, x2, x3, x4가 training_iris 각 종의 x1, x2, x3, x4와 일치하는 수
    for i in range(30):
        x1 = test_iris.iloc[i, 0]
        x2 = test_iris.iloc[i, 1]
        x3 = test_iris.iloc[i, 2]
        x4 = test_iris.iloc[i, 3]
        x5 = test_iris.iloc[i, 4]
        test_arr = np.array([x1, x2, x3, x4])

        # 각 종의 likelihood
        # Multi-variate Gaussian distribution pdf
        z_s_arr = test_arr - mean_s_arr
        z_v_arr = test_arr - mean_v_arr
        z_vi_arr = test_arr - mean_vi_arr
        pdf_s = (np.exp((-1 / 2) * np.dot(np.dot(np.transpose(z_s_arr), np.linalg.inv(var_s_arr)), z_s_arr)) /
                 np.sqrt((2 * np.pi) ** 2 * np.linalg.det(var_s_arr)))
        pdf_v = (np.exp((-1 / 2) * np.dot(np.dot(np.transpose(z_v_arr), np.linalg.inv(var_v_arr)), z_v_arr)) /
                 np.sqrt((2 * np.pi) ** 2 * np.linalg.det(var_v_arr)))
        pdf_vi = (np.exp((-1 / 2) * np.dot(np.dot(np.transpose(z_vi_arr), np.linalg.inv(var_vi_arr)), z_vi_arr)) /
                  np.sqrt((2 * np.pi) ** 2 * np.linalg.det(var_vi_arr)))
        # 각 종의 prior
        per_s = ts / 120
        per_v = tv / 120
        per_vi = tvi / 120
        # 각 종의 posterior
        pos_s = (per_s * pdf_s) / (per_s * pdf_s + per_v * pdf_v + per_vi * pdf_vi)
        pos_v = (per_v * pdf_v) / (per_s * pdf_s + per_v * pdf_v + per_vi * pdf_vi)
        pos_vi = (per_vi * pdf_vi) / (per_s * pdf_s + per_v * pdf_v + per_vi * pdf_vi)
        # classifier 결과와 tp 카운트
        if max(pos_s, pos_v, pos_vi) == pos_s:
            s_cnt += 1
            if x5 == 'Iris-setosa':
                s_tp += 1
        elif max(pos_s, pos_v, pos_vi) == pos_v:
            v_cnt += 1
            if x5 == 'Iris-versicolor':
                v_tp += 1
        else:
            vi_cnt += 1
            if x5 == 'Iris-virginica':
                vi_tp += 1
    # 각 test precision, recall
    test_precision = s_tp / s_cnt + v_tp / v_cnt + vi_tp / vi_cnt
    test_recall = s_tp / s + v_tp / v + vi_tp / vi
    test_precision /= 3
    test_recall /= 3
    # 최종 precision, recall 구하기 위한 global 변수
    global precision, recall
    precision += test_precision
    recall += test_recall
    tab = '    '
    print(s_cnt, v_cnt, vi_cnt, tab, s_tp, v_tp, vi_tp, tab, s, v, vi, tab, test_precision, tab, test_recall, '\n')


print('\n------------------------------------------------------------------------------------')
posterior(training1, test1)
posterior(training2, test2)
posterior(training3, test3)
posterior(training4, test4)
posterior(training5, test5)
print('------------------------------------------------------------------------------------')
print(precision / 5, '        ', recall / 5)
print('------------------------------------------------------------------------------------\n')
