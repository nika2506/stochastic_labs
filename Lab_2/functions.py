import numpy as np
import matplotlib.pyplot as plt
import math


def calculateZones(data):
    count = []
    start = []
    finish = []
    bin = int(math.log2(len(data) + 1))
    hist = plt.hist(data, bins=bin)
    plt.show()
    types = [0] * bin
    for i in range(bin):
        count.append(hist[0][i])
        start.append(hist[1][i])
        finish.append(hist[1][i + 1])
    sorted_hist = sorted(count)
    repeat = 0
    for i in range(bin):
        for j in range(bin):
            if sorted_hist[len(sorted_hist) - 1 - i] == count[j]:
                if repeat == 0:
                    types[j] = "background"
                elif repeat == 1:
                    types[j] = "data"
                else:
                    types[j] = "transition"
                repeat += 1
    return start, finish, types


def convertData(data, indices, start, end, types):
    size_ind = len(indices)
    point_types = [0] * size_ind
    zones = []
    type_zones = []
    for i in range(size_ind):
        for j in range(len(types)):
            if (data[i] >= start[j]) and (data[i] <= end[j]):
                point_types[i] = types[j]
    curr_type = point_types[0]
    start = 0
    for i in range(len(point_types)):
        if curr_type != point_types[i]:
            end = i
            type_zones.append(curr_type)
            zones.append([start, end])
            start = end
            curr_type = point_types[i]
    if curr_type != type_zones[len(type_zones) - 1]:
        type_zones.append(curr_type)
        zones.append([end, size_ind - 1])
    return zones, type_zones


def smoothSignal(data, indices, zones, types):
    while len(types) > 5:
        for i in range(len(types)):
            if (types[i] == "transition") and (types[i - 1] == types[i + 1]):
                start_val = data[zones[i - 1][1] - 1]
                finish_val = data[zones[i + 1][0] + 1]
                new_val = (start_val + finish_val) / 2
                num = zones[i][1] - zones[i][0]
                for k in range(num):
                    data[zones[i][0] + k] = new_val
        start, finish, types = calculateZones(data)
        zones, types = convertData(data, indices, start, finish, types)
    return data, zones, types


def makePlotWithZones(signal, indices, zones, types):
    size = len(zones)
    for i in range(size):
        if types[i] == "background":
            color_ = 'y'
        if types[i] == "data":
            color_ = 'r'
        if types[i] == "transition":
            color_ = 'g'
        plt.plot(indices[zones[i][0]:zones[i][1]],
                 signal[zones[i][0]:zones[i][1]], color=color_, label=types[i])
    plt.legend()
    plt.show()


def intergroupDispersion(data):
    means_array = np.mean(data, axis=1)
    mean = np.mean(means_array)
    sum = np.sum((means_array - mean) ** 2)
    disp = len(data) * (sum / (data.shape[0] - 1))
    return disp


def intragroupDispersion(data):
    disp = 0
    size = len(data)
    for i in range(size):
        sum = np.sum((data[i] - np.mean(data[i])) ** 2) / (size - 1)
        disp += sum / size
    return disp


def applayFisher(data, k):
    newSizeY = int(len(data) / k)
    newSizeX = k
    print("intervals: " + str(k))
    splitData = np.reshape(data, (newSizeX, newSizeY))
    inter_disp = intergroupDispersion(splitData)
    print("intergroupDispersion: " + str(inter_disp))
    intra_disp = intragroupDispersion(splitData)
    print("intragroupDispersion: " + str(intra_disp))
    fisher_criteria = inter_disp / intra_disp
    print("FisherCriteria: " + str(fisher_criteria))
    return fisher_criteria


def findNearestValue(num):
    i = 4
    while num % i != 0:
        i += 1
    return i


def calculateFisherCriteria(data, zones_data):
    fisher_criteria = []
    for i in range(len(zones_data)):
        start = zones_data[i][0]
        finish = zones_data[i][1]
        m = findNearestValue(finish - start)
        while m == finish - start:
            finish += 1
            m = findNearestValue(finish - start)
        fisher_criteria.append(applayFisher(data[start:finish], int(m)))
    return fisher_criteria