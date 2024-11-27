import pandas as pd

# Lista de los archivos CSV
file_count = 15
filenames = [f"results_f{i}.csv" for i in range(1, file_count + 1)]

# Filas de interés
rows_of_interest = [120000, 600000, 2700000]

# Crear una lista para almacenar los DataFrames de cada archivo
columns = []

# Leer cada archivo
for i, filename in enumerate(filenames, start=1):
    try:
        # Leer el archivo como DataFrame
        df = pd.read_csv(filename, header=None, names=["X", "Y", "Z"])
        
        # Filtrar solo las filas de interés
        filtered = df[df["X"].isin(rows_of_interest)]
        
        # Reiniciar los índices y renombrar la columna Z
        filtered = filtered.reset_index(drop=True)
        filtered = filtered[["Z"]].rename(columns={"Z": f"f{i}"})
        
        # Agregar columna al resultado
        columns.append(filtered)
    except Exception as e:
        print(f"Error leyendo el archivo {filename}: {e}")

# Combinar todas las columnas en un único DataFrame
if columns:
    result_df = pd.concat(columns, axis=1)
    
    # Generar el índice en el orden deseado
    num_repeats = len(result_df) // len(rows_of_interest)
    row_labels = rows_of_interest * num_repeats
    result_df["Row"] = row_labels
    result_df.sort_index(inplace=True)  # Asegurar el orden
    result_df.set_index("Row", inplace=True)
    
    # Guardar la tabla en un archivo CSV
    result_df.to_csv("filtered_table.csv")
    print("Tabla creada y guardada como 'filtered_table.csv'")
else:
    print("No se encontraron datos para concatenar.")
