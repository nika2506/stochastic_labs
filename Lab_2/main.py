import functions as func
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    size_set = 1024  # кол-во чисел в одном наборе
    num_set = 323  # номер набора, который мы хотим взять для дальнейшей обработки

    ind_array = [i for i in range(size_set)]  # массив индексов
    data_full = np.loadtxt("wave_ampl.txt", delimiter=', ', dtype=np.float)  # массив со всеми наборами чисел из файла
    size_sets = len(data_full) / size_set  # кол-во наборов
    first_ind = num_set * size_set  # индекс в массиве data_full, с которого начинается выбранный нами набор
    data = np.zeros(size_set)  # один выбранный нами набор чисел
    for i in range(size_set):
        data[i] = data_full[first_ind + i]

    plt.title(num_set)
    plt.plot(data)
    plt.show()
    num_set += 1

    start, finish, types = func.calculateZones(data)
    zones, types = func.convertData(data, ind_array, start, finish, types)

    copy_data = data.copy()
    smoothed_signal, zones, types = func.smoothSignal(copy_data, ind_array, zones, types)
    plt.title(" Without Emissions ")
    plt.plot(smoothed_signal)
    plt.show()
    func.makePlotWithZones(data, ind_array, zones, types)

    fisher_criteria = func.calculateFisherCriteria(data, zones)

