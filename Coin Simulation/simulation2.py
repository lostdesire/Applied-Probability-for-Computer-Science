import matplotlib.pyplot as plt
import numpy as np

# problem 1-b
# n = 100, p(HEAD) = 0.6
n = 100
H_cnt = 0
H_proportion = [0.0 for i in range(n)]
for i in range(0, n):
    # 0 = TAIL / 1 = HEAD
    sample = np.random.choice([0, 1], p=[0.4, 0.6])
    H_cnt += sample
    H_proportion[i] = H_cnt / (i + 1)
    if sample == 1:
        print('H', H_proportion[i])
    else:
        print('T', H_proportion[i])

x = range(1, n+1)
y = [H_proportion[j - 1] for j in x]
plt.plot(x, y)

plt.show()
