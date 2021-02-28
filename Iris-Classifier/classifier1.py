import pandas as pd
import sklearn as sk

# Iris : classifier (likelihood pmf)

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


# classifier 분류 결과와, tp, test data 실제 종 수
# precision, recall 값
def posterior(training_iris, test_iris):
    s_cnt, v_cnt, vi_cnt = 0, 0, 0
    s_tp, v_tp, vi_tp = 0, 0, 0

    # 각 종의 training data 수
    ts = len(training_iris[training_iris.species == 'Iris-setosa'])
    tv = len(training_iris[training_iris.species == 'Iris-versicolor'])
    tvi = len(training_iris[training_iris.species == 'Iris-virginica'])
    # 각 종의 test data 수
    s = len(test_iris[test_iris.species == 'Iris-setosa'])
    v = len(test_iris[test_iris.species == 'Iris-versicolor'])
    vi = len(test_iris[test_iris.species == 'Iris-virginica'])

    # iris training
    for i in range(30):
        x1 = test_iris.iloc[i, 0]
        x2 = test_iris.iloc[i, 1]
        x3 = test_iris.iloc[i, 2]
        x4 = test_iris.iloc[i, 3]
        x5 = test_iris.iloc[i, 4]
        sx1_cnt, sx2_cnt, sx3_cnt, sx4_cnt = 0, 0, 0, 0
        vx1_cnt, vx2_cnt, vx3_cnt, vx4_cnt = 0, 0, 0, 0
        vix1_cnt, vix2_cnt, vix3_cnt, vix4_cnt = 0, 0, 0, 0

        # test_iris x1, x2, x3, x4가 training_iris 각 종의 x1, x2, x3, x4와 일치하는 수
        for j in range(120):
            if training_iris.iloc[j, 0] == x1:
                if training_iris.iloc[j, 4] == 'Iris-setosa':
                    sx1_cnt += 1
                elif training_iris.iloc[j, 4] == 'Iris-versicolor':
                    vx1_cnt += 1
                else:
                    vix1_cnt += 1
            if training_iris.iloc[j, 1] == x2:
                if training_iris.iloc[j, 4] == 'Iris-setosa':
                    sx2_cnt += 1
                elif training_iris.iloc[j, 4] == 'Iris-versicolor':
                    vx2_cnt += 1
                else:
                    vix2_cnt += 1
            if training_iris.iloc[j, 2] == x3:
                if training_iris.iloc[j, 4] == 'Iris-setosa':
                    sx3_cnt += 1
                elif training_iris.iloc[j, 4] == 'Iris-versicolor':
                    vx3_cnt += 1
                else:
                    vix3_cnt += 1
            if training_iris.iloc[j, 3] == x4:
                if training_iris.iloc[j, 4] == 'Iris-setosa':
                    sx4_cnt += 1
                elif training_iris.iloc[j, 4] == 'Iris-versicolor':
                    vx4_cnt += 1
                else:
                    vix4_cnt += 1

        # 각 종의 likelihood
        # laplace smoothing
        like_s = ((sx1_cnt + 1) * (sx2_cnt + 1) * (sx3_cnt + 1) * (sx4_cnt + 1) / (ts + 1) ** 4)
        like_v = ((vx1_cnt + 1) * (vx2_cnt + 1) * (vx3_cnt + 1) * (vx4_cnt + 1) / (tv + 1) ** 4)
        like_vi = ((vix1_cnt + 1) * (vix2_cnt + 1) * (vix3_cnt + 1) * (vix4_cnt + 1) / (tvi + 1) ** 4)
        # 각 종의 prior
        pri_s = ts / 120
        pri_v = tv / 120
        pri_vi = tvi / 120
        # 각 종의 posterior
        pos_s = (pri_s * like_s) / (pri_s * like_s + pri_v * like_v + pri_vi * like_vi)
        pos_v = (pri_v * like_v) / (pri_s * like_s + pri_v * like_v + pri_vi * like_vi)
        pos_vi = (pri_vi * like_vi) / (pri_s * like_s + pri_v * like_v + pri_vi * like_vi)
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
print(precision/5, '        ', recall/5)
print('------------------------------------------------------------------------------------\n')
