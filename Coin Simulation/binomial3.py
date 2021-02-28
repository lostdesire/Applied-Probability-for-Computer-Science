import matplotlib.pyplot as plt


# function of nCr
def ncr(a, b):
    if a < 1 or b < 0 or a < b:
        return 0
    b = min(b, a - b)
    num = 1
    den = 1
    for i in range(0, b):
        num *= (a - i)
        den *= (i + 1)
    return num / den


# problem 2-d
# n = 1000, p(HEAD) = 0.01, binomial sample = 500
# pmf to compare Poisson distribution lambda = 10
n = 1000
p = 0.01
sample = 500
H_cnt = 0
pk_cnt = [0 for i in range(n + 1)]
for j in range(0, sample):
    for i in range(0, n + 1):
        H_cnt = ncr(n, i) * p ** i * (1 - p) ** (n - i)
        pk_cnt[i] += H_cnt

for i in range(0, 25):
    print(pk_cnt[i])
x = range(0, n+1)
y = [pk_cnt[j] / 500 for j in x]
plt.plot(x, y, 'bo-')
plt.xlim([0, 20])
plt.ylim([0, 0.40])

plt.show()
