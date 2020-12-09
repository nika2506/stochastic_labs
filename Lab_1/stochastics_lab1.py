from matplotlib import pylab as plt
import math
import numpy as np
import numpy.random as rand
import random
import scipy.stats as stats

num = 100  # кол-во элементов в выборке
hist_num = 10  # кол-во столбцов на гистограмме
#mu, sigma = 0, 0.43  # среднее значение и стандартное отклонение
A, B = -2, 2  # интервал от A до B

normal_distrAB = np.zeros((num))
k = 0
while k < num: #генерируем нормально распределенные величины в интервале [A, B]
    #rand_num = random.gauss(mu, sigma)
    rand_num = rand.standard_normal()
    if A < rand_num and rand_num < B:
        normal_distrAB[k] = rand_num
        k += 1
normal_countAB, normal_binsAB, normal_d = plt.hist(normal_distrAB, hist_num)
plt.show()

uniform_distr = rand.uniform(A, B, num) #генерация равномерно распределенных величин
uniform_count, uniform_bins, uniform_ignored = plt.hist(uniform_distr, hist_num)
expected_uniform_count = np.zeros_like(uniform_count) + (1/hist_num) * num
plt.show()

expected_normal_count = np.zeros_like(normal_countAB)
for i in range(hist_num):
    expected_normal_count[i] = (stats.norm.cdf(normal_binsAB[i + 1]) - stats.norm.cdf(normal_binsAB[i])) * num

#x2_nn_arr = np.zeros(hist_num)
#x2_nu_arr = np.zeros(hist_num)
x2_uu = 0 #критерий, проверяющий соответствие равномерной выборки равномерному распределению
x2_un = 0 #критерий, проверяющий соответствие равномерной выборки нормальному распределению
x2_nu = 0 #критерий, проверяющий соответствие нормальной выборки равномерному распределению
x2_nn = 0 #критерий, проверяющий соответствие нормальной выборки нормальному распределению
for i in range(hist_num):
        x2_uu += (uniform_count[i] - expected_uniform_count[i]) ** 2 / expected_uniform_count[i]
        x2_un += (uniform_count[i] - expected_normal_count[i]) ** 2 / expected_normal_count[i]
        x2_nu += (normal_countAB[i] - expected_uniform_count[i]) ** 2 / expected_uniform_count[i]
        x2_nn += (normal_countAB[i] - expected_normal_count[i]) ** 2 / expected_normal_count[i]
        #x2_nn_arr[i] = (normal_countAB[i] - expected_normal_count[i]) ** 2 / expected_normal_count[i]
        #x2_nu_arr[i] = (normal_countAB[i] - expected_uniform_count[i]) ** 2 / expected_uniform_count[i]

print(x2_uu)
print(x2_un)
print(x2_nu)
print(x2_nn)

#для заполнения таблиц
'''print("------------")
print("ni: ", normal_countAB, "sum_ni: ", np.sum(normal_countAB))
print("pi: ", expected_normal_count/num, "sum_pi: ", np.sum(expected_normal_count/num))
print("npi: ", expected_normal_count, "sum_npi: ", np.sum(expected_normal_count))
print("ni - npi: ", normal_countAB - expected_normal_count, "sum_ni-npi: ", np.sum(normal_countAB - expected_normal_count))
print("X2: ", x2_nn_arr)

print("------------")
print("ni: ", normal_countAB, "sum_ni: ", np.sum(normal_countAB))
print("pi: ", expected_uniform_count/num, "sum_pi: ", np.sum(expected_uniform_count/num))
print("npi: ", expected_uniform_count, "sum_npi: ", np.sum(expected_uniform_count))
print("ni - npi: ", normal_countAB - expected_uniform_count, "sum_ni-npi: ", np.sum(normal_countAB - expected_uniform_count))
print("X2: ", x2_nu_arr)'''
