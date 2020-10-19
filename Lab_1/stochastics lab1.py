from matplotlib import pylab as plt
import math
import numpy as np
import numpy.random as rand
import random

num = 500  # кол-во элементов в выборке
hist_num = 20  # кол-во столбцов на гистограмме
mu, sigma = 0, 0.44  # среднее значение и стандартное отклонение
A, B = -0.5, 0.5  # интервал от A до B

normal_distrAB = np.zeros((num))
k = 0
while k < num: #генерируем нормально распределенные величины в интервале [A, B]
    rand_num = random.gauss(mu, sigma)
    if A < rand_num and rand_num < B:
        normal_distrAB[k] = rand_num
        k += 1

normal_countAB, normal_binsAB, normal_distrAB = plt.hist(normal_distrAB, hist_num, density=True)
plt.plot(normal_binsAB, 1 / (sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (normal_binsAB - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
plt.show()

uniform_distr = rand.uniform(A, B, num) #генерация равномерно распределенных величин
uniform_count, uniform_bins, uniform_ignored = plt.hist(uniform_distr, hist_num, density=True)
expected_uniform_count = np.zeros_like(uniform_count) + 1/(B-A)
plt.plot(uniform_bins, np.zeros_like(uniform_bins) + np.mean(uniform_count), linewidth=2, color='y')
plt.show()

expected_normal_count = np.zeros_like(normal_countAB)
for i in range(hist_num):
    x = (normal_binsAB[i] + normal_binsAB[i+1])/2
    expected_normal_count[i] = math.exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) * (1/(sigma * math.sqrt(2*math.pi)))

x2_uu = 0 #критерий, проверяющий соответствие равномерной выборки равномерному распределению
x2_un = 0 #критерий, проверяющий соответствие равномерной выборки нормальному распределению
x2_nu = 0 #критерий, проверяющий соответствие нормальной выборки равномерному распределению
x2_nn = 0 #критерий, проверяющий соответствие нормальной выборки нормальному распределению
for i in range(hist_num):
        x2_uu += (uniform_count[i] - expected_uniform_count[i]) * (uniform_count[i] - expected_uniform_count[i]) / expected_uniform_count[i]
        x2_un += (uniform_count[i] - expected_normal_count[i]) * (uniform_count[i] - expected_normal_count[i]) / \
                 expected_normal_count[i]
        x2_nu += (normal_countAB[i] - expected_uniform_count[i]) * (normal_countAB[i] - expected_uniform_count[i]) / expected_uniform_count[i]
        x2_nn += (normal_countAB[i] - expected_normal_count[i]) * (normal_countAB[i] - expected_normal_count[i]) / expected_normal_count[i]

print(x2_uu)
print(x2_un)
print(x2_nu)
print(x2_nn)
