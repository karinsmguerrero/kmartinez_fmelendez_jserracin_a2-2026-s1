import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# =========================
# LECTURA DEL CSV
# =========================

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "perf_results.csv"

def load_csv(path: Path):
    grupos = defaultdict(lambda: defaultdict(list))

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                nbodies = int(row["nbodies"])
                program = row["program"].strip()
                time_sec = float(row["time_sec"])
            except (KeyError, ValueError):
                continue

            grupos[nbodies][program].append(time_sec)

    return grupos

# =========================
# FUNCIONES ESTADÍSTICAS
# =========================

def arithmetic_mean(data):
    return np.mean(data)

def median(data):
    return np.median(data)

def geometric_mean(data):
    return np.exp(np.mean(np.log(np.array(data, dtype=float))))

def confidence_interval(data, confidence=0.95):
    data = np.array(data, dtype=float)
    n    = len(data)
    mean = np.mean(data)
    std  = np.std(data, ddof=1)
    h    = stats.t.ppf((1 + confidence) / 2, n - 1) * (std / np.sqrt(n))
    return mean, h

def speedup_gm(seq, other):
    """Speedup basado en medias geométricas: gm(seq) / gm(other)."""

    return float(geometric_mean(seq) / geometric_mean(other))

# =========================
# GUARDAR RESULTADOS
# =========================

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "resultados.csv"

FIELDNAMES = [
    "nbodies",
    "program",
    "media_aritmetica",
    "mediana",
    "desviacion_estandar",
    "media_geometrica",
    "IC",
    "SpeedUp",
    "Eficiencia",
]

PARALLEL_UNITS = 8 # esto es solo para el de simd, hay que ver despues como se hace para el de gpu

def calcular_fila(nbodies, program: str, times_sec, seq_times=None):
    times_sec = np.array(times_sec, dtype=float)
    am = arithmetic_mean(times_sec)
    med = median(times_sec)
    std = float(np.std(times_sec, ddof=1)) if len(times_sec) > 1 else 0.0
    gm = geometric_mean(times_sec)
    _, ci = confidence_interval(times_sec)

    if seq_times is not None and program != "n-body":
        n_min = min(len(seq_times), len(times_sec))
        sp = speedup_gm(np.array(seq_times[:n_min]), times_sec[:n_min])
    else:
        sp = 1.0

    eff = 0.0 if program == "n-body" else (float(sp) / PARALLEL_UNITS) if program == "cpu_simd" else (float(sp) / 256) 

    return {
        "nbodies": nbodies,
        "program":             program,
        "media_aritmetica":    round(float(am), 9),
        "mediana":             round(float(med), 9),
        "desviacion_estandar": round(float(std), 9),
        "media_geometrica":    round(float(gm), 9),
        "IC":                  round(float(ci), 9),
        "SpeedUp":             round(float(sp), 6),
        "Eficiencia":          round(float(eff), 6),
    }

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    datos = load_csv(CSV_PATH)

    filas = []

    for nbodies in sorted(datos.keys()):

        grupos = datos[nbodies]

        seq_times = grupos.get("n-body", [])
        if not seq_times:
            print(f"Advertencia: no hay baseline para {nbodies}")
            continue

        prefer_order = ["n-body", "cpu_simd", "build/n-body_gpu"]

        programs = list(grupos.keys())
        programs.sort(
            key=lambda p: (
                prefer_order.index(p)
                if p in prefer_order
                else 999,
                p,
            )
        )

        for program in programs:
            filas.append(
                calcular_fila(
                    nbodies,
                    program,
                    grupos[program],
                    seq_times=seq_times,
                )
            )