import pandas as pd
import openpyxl
import numpy as np
# extract from csv and save to the database
# estaria de huevos que del mismo script pueda transformar o entender el csv antes de extraerlo.

def extract_adjacency_matrix(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Extract the adjacency matrix
    adjacency_matrix = df.values
    
    return adjacency_matrix

def process_first_column(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Extract text from the first column and delete the first word
    first_column = df.iloc[:, 0].apply(lambda x: ' '.join(str(x).split(' ')[1:]))
    
    return first_column

def extract_adjacency_matrix(file_path):

    first_column_processed = process_first_column(file_path)
    print(first_column_processed.dtypes)
    print(first_column_processed)

    # ...existing code...
    # parsing for words and dictionary for word counting
    df = np.array(first_column_processed)
    print(df)

    countDict = {}
    for competencia in df:
        competencia = competencia.split()
        for word in competencia:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1

    print(countDict)

    # Construct adjacency matrix dataframe for words shared
    competences = df.tolist()
    adjacency_matrix = pd.DataFrame(0, index=competences, columns=competences)

    for i, comp1 in enumerate(competences):
        words1 = set(comp1.split())
        for j, comp2 in enumerate(competences):
            if i != j:
                words2 = set(comp2.split())
                if words1.intersection(words2):
                    adjacency_matrix.iloc[i, j] = 1

    print(adjacency_matrix)
    return adjacency_matrix


file_path = 'arquitectura1.xlsx'
extract_adjacency_matrix(file_path)