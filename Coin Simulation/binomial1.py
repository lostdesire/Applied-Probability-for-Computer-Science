import matplotlib.pyplot as plt


# problem 2-a
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


# n = 10, p(HEAD) = 0.5, binomial sample = 1024
n = 10
p = 0.5
sample = 1024
H_cnt = 0
pk_cnt = [0 for i in range(n + 1)]
for j in range(0, sample):
    for i in range(0, n + 1):
        H_cnt = ncr(n, i) * p ** i * (1 - p) ** (n - i)
        pk_cnt[i] += H_cnt

for i in range(0, n + 1):
    print(pk_cnt[i])
x = range(0, n+1)
y = [pk_cnt[j] for j in x]
plt.bar(x, y)

plt.show()
