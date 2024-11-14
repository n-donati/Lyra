import pandas as pd

def process_row(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Extract only the first row of column 'CE'
    # Since pandas uses zero-based indexing, ensure 'CE' is the correct label from your preview data.
    data_in_ce = df.iloc[:, 64]  # Accessing data using column label 'CE' and row index 0 for the first row.
    calificaciones = df.iloc[:, 65]
    
    # Delete the first word from the cell in column 'CE'
    # Convert the cell to string, split it, and join it back excluding the first word
    processed_data = data_in_ce.apply(lambda x: ' '.join(str(x).split()[1:]) if pd.notna(x) and isinstance(x, str) else x)
    
    # print("Processed Data from column CE:\n", processed_data)
    return processed_data


def extract_adjacency_matrix(file_path):
    first_row_processed = process_row(file_path)

    # Convert the processed data to a list, then to an array
    processed_list = first_row_processed.tolist()
    # print("Processed List:", processed_list)

    # Parse for words and create a dictionary for word counting
    countDict = {}
    for item in processed_list:
        words = item.split()
        for word in words:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1

    # print("Word Count Dictionary:", countDict)

    # Construct adjacency matrix dataframe for words shared
    competences = processed_list
    adjacency_matrix = pd.DataFrame(0, index=competences, columns=competences)

    for i, comp1 in enumerate(competences):
        words1 = set(comp1.split())
        for j, comp2 in enumerate(competences):
            if i != j and words1.intersection(set(comp2.split())):
                adjacency_matrix.iloc[i, j] = 1

    # print("Adjacency Matrix:\n", adjacency_matrix)
    return adjacency_matrix

# Assuming the correct file path
# file_path = 'calificaciones1.xlsx'
# extract_adjacency_matrix(file_path)
