#!/usr/bin/env python3

import json
import glob
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

# Rutas basadas en la ubicación del script (más robusto que rutas relativas al CWD)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, ".."))
profiling_dir = os.path.join(base_dir, "profiling")

# Directorio donde se guardarán las gráficas
output_dir = os.path.join(profiling_dir, "metric_plots")
os.makedirs(output_dir, exist_ok=True)

# Almacenar todos los datos
data = []

metric_names = ["GPU Memory Usage [0] (S)",
"GPU MeanOccupancyPerCU [0]",
"GPU MeanOccupancyPerActiveCU [0]",
"GPU L2CacheHit [0]",
"GPU GRBM_COUNT [0]",
"GPU GL2C_MISS [0]"]

metric_titles = ["GPU Memory Usage (MB)",
"GPU Mean Occupancy Per CU",
"GPU Mean Occupancy Per Active CU",
"GPU L2 Cache Hit",
"GPU GRBM COUNT",
"GPU GL2C MISS"]

# Leer archivos results_XXXX.json (desde el directorio de profiling)
pattern = os.path.join(profiling_dir, "**", "results_*.json")
files = glob.glob(pattern, recursive=True)
if not files:
    print(f"No se encontraron archivos: {pattern}")

for filename in files:
    match = re.search(r"results_(\d+)\.json", filename)
    if not match:
        continue

    nbodies = int(match.group(1))

    with open(filename, "r") as f:
        metrics = json.load(f)

    for metric in metrics:
        if(metric["Name"] in metric_names):
            try:
                data.append({
                    "nbodies": nbodies,
                    "Name": metric["Name"],
                    "Max value": float(metric["Max value"])
                })
            except (KeyError, ValueError):
                continue

# Crear DataFrame
df = pd.DataFrame(data)

if df.empty:
    print("No se encontraron métricas para graficar. Asegúrate de que los JSON contienen datos válidos.")
    exit(0)

# Generar una gráfica por métrica
for metric_name, group in df.groupby("Name"):

    group = group.sort_values("nbodies")

    plt.figure(figsize=(8, 5))
    plt.plot(
        group["nbodies"],
        group["Max value"],
        marker="o"
    )

    plt.title(metric_titles[metric_names.index(metric_name)])
    plt.xlabel("Number of Bodies")
    plt.ylabel("Max Value")
    plt.grid(True)

    # Limpiar nombre para archivo
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '_', metric_name)
    output_file = os.path.join(output_dir, f"{safe_name}.png")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")