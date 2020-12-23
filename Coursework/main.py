import pandas as pd
from sklearn.decomposition import PCA
import functions as func


if __name__ == '__main__':
    sheet_1, sheet_2, sheet_3 = func.getSheetsFromExelFile('data (changed).xlsx')  # считывание из exel файла 3-ех первых страниц
    sheet_1 = func.removeExtraColumns(sheet_1)  # удилим столбцы '№ п.п' и 'проба'
    sheet_2 = func.removeExtraColumns(sheet_2)
    sheet_3 = func.removeExtraColumns(sheet_3)
    elements = sheet_1.columns.values # отдельно сохраним названия столбцов (веществ)
    num_elements = len(elements)

    sheet_1 = func.fillTheGaps(sheet_1)  # заполним пропуски в таблицах
    sheet_2 = func.fillTheGaps(sheet_2)
    sheet_3 = func.fillTheGaps(sheet_3)
    sheet_1 = func.smoothOutEmissions(sheet_1) # сгладим выбросы в таблицах
    sheet_2 = func.smoothOutEmissions(sheet_2)
    sheet_3 = func.smoothOutEmissions(sheet_3)

    all_sheets = sheet_1.append(sheet_2) # объединим таблицы
    all_sheets = all_sheets.append(sheet_3)
    all_sheets = func.scaleData(all_sheets)

    size_sheet_1 = len(sheet_1)
    size_sheet_2 = size_sheet_1 + len(sheet_2)

    pca = PCA(n_components=num_elements)
    pcs_df = pd.DataFrame(pca.fit_transform(all_sheets))

    func.showHistPCA(pca)
    func.showSamplesPlot(pcs_df, size_sheet_1, size_sheet_2)

    pca1, pca2, pca3 = func.calculatePCAs(sheet_1, sheet_2, sheet_3, num_elements)
    func.calculateAndPrintContributions(pca1, pca2, pca3, pca, elements)
