import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import mstats
from sklearn.decomposition import PCA

def getSheetsFromExelFile(path):
    file = pd.ExcelFile(path)
    sheets = {name: file.parse(name) for name in file.sheet_names}
    return sheets['1'], sheets['2'], sheets['3']


def removeExtraColumns(df):
    return df.drop(columns=['№ п.п', 'проба'])


def fillTheGaps(df):
    df_with_nan = df.replace(0, np.nan)
    df_with_nan = df_with_nan.replace('-', np.nan)
    return df_with_nan.fillna(df_with_nan.mean())


def scaleData(df):
    return pd.DataFrame(StandardScaler().fit_transform(df))


def smoothOutEmissions(df):
    new_df = df.copy()
    num_col = new_df.shape[1] - 1
    for i in range(num_col):
        col_name = new_df.columns[i]
        column = new_df[col_name].values
        mstats.winsorize(column, limits=[0.2, 0.2], inplace=True)
    return new_df


def showSamplesPlot(PCs_df, size1, size2):
    plt.scatter(x=PCs_df.iloc[:size1, 0], y=PCs_df.iloc[:size1, 1], c='b', label='выборка №1')
    plt.scatter(x=PCs_df.iloc[size1:size2, 0], y=PCs_df.iloc[size1:size2, 1], c='r', label='выборка №2')
    plt.scatter(x=PCs_df.iloc[size2:, 0], y=PCs_df.iloc[size2:, 1], c='g', label='выборка №3')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


def showHistPCA(pca):
    num = range(pca.n_components_)
    plt.bar(num, pca.explained_variance_ratio_)
    plt.xlabel('Компоненты')
    plt.ylabel('Объясненная дисперсия')
    plt.xticks(num)
    plt.show()


def calculateContributions(pca, idx, features):
    PC = np.array(pca.components_[idx])
    tmp_idxs = np.argsort(np.abs(PC))[::-1]
    return np.array(features)[tmp_idxs], PC[tmp_idxs]


def printContributions(elem, coef):
    contributions = pd.DataFrame.from_dict({'Элемент': elem, 'Вклад': coef})
    print(contributions)


def calculatePCAs(data1, data2, data3, size):
    pca1 = PCA(n_components=size).fit(data1)
    pca2 = PCA(n_components=size).fit(data2)
    pca3 = PCA(n_components=size).fit(data3)
    return pca1, pca2, pca3


def calculateAndPrintContributions(pca1, pca2, pca3, pca, elements):
    print("sheet 1")
    features1, coeffs1 = calculateContributions(pca1, 0, elements)
    printContributions(features1, coeffs1)
    print("sheet 2")
    features2, coeffs2 = calculateContributions(pca2, 0, elements)
    printContributions(features2, coeffs2)
    print("sheet 3")
    features3, coeffs3 = calculateContributions(pca3, 0, elements)
    printContributions(features3, coeffs3)
    print("sheet 1,2,3")
    features, coeffs = calculateContributions(pca, 0, elements)
    printContributions(features, coeffs)