import pandas as pd
import os

# Directorios de los algoritmos
algorithms = ["shade", "ERDGshade", "shadeils", "ERDGshadeils"]

# Lista para acumular todos los datos promediados
all_data = []

# Para cada archivo results_f$i.csv (del 1 al 15)
for i in range(1, 16):
    for algo in algorithms:
        file_path = os.path.join(algo, f"results_f{i}.csv")
        
        # Leer el archivo csv con todas las ejecuciones
        data = pd.read_csv(file_path, header=None, names=["Iterations", "Metric1", "Value"])
        
        # Calcular la media de `Value` en cada punto de iteración
        mean_data = data.groupby("Iterations")["Value"].mean().reset_index()
        mean_data["Algorithm"] = algo  # Agregar columna del algoritmo
        mean_data["FileID"] = i  # Agregar columna del ID del archivo
        
        all_data.append(mean_data)

# Concatenar todos los datos promediados en un solo DataFrame y exportar a CSV
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("taco_results.csv", index=False)