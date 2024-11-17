import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

# Directorios de los algoritmos
algorithms = ["shade", "ERDGshade", "shadeils", "ERDGshadeils"]

# Para cada archivo results_f$i.csv (del 1 al 15)
for i in range(1, 16):
    plt.figure(figsize=(10, 6))

    # Leer y graficar cada archivo
    for algo in algorithms:
        file_path_pattern = os.path.join(algo, f"results_f{i}.csv")
        
        # Leer todas las ejecuciones en el archivo
        data = pd.read_csv(file_path_pattern, header=None, names=["Iterations", "Metric1", "Value"])
        
        # Calcular la media de `Value` en cada punto de iteración
        mean_data = data.groupby("Iterations")["Value"].mean().reset_index()

        # Graficar la media de `Value` en función de 'Iterations'
        plt.plot(mean_data["Iterations"], mean_data["Value"], label=algo)

    plt.xlabel("Iterations")
    plt.ylabel("Average Value")
    plt.title(f"Comparison of Algorithms for results_f{i} (Average of 25 runs)")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")

    # Guardar la gráfica como imagen
    plt.savefig(f"comparison_results_f{i}.png")
    plt.close()
