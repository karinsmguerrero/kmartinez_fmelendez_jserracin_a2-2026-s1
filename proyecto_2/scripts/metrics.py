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
    """Lee `perf_results.csv`.

    Espera columnas: run, program, time_sec

    Devuelve: dict[str, list[float]] donde la llave es `program` y el valor
    es una lista de tiempos (segundos).
    """

    # grupos[(program, nbodies)] -> [times]
    grupos: dict[tuple, list[float]] = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prog = (row.get("program") or "").strip()
            t = (row.get("time_sec") or "").strip()
            nb = (row.get("nbodies") or "").strip()

            if not prog or not t or not nb:
                continue

            try:
                grupos[(prog, int(nb))].append(float(t))
            except ValueError:
                continue

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
    "program",
    "nbodies",
    "media_aritmetica",
    "mediana",
    "desviacion_estandar",
    "media_geometrica",
    "IC",
    "SpeedUp",
    "Eficiencia",
]

PARALLEL_UNITS = 8 # simd
GPU_PARALLEL_UNITS = 64 # gpu

def calcular_fila(program: str, times_sec, seq_times=None):
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

    eff = 0.0 if program == "n-body" else (float(sp) / PARALLEL_UNITS) if program == "cpu_simd" else (float(sp) / GPU_PARALLEL_UNITS) 

    return {
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
    grupos = load_csv(CSV_PATH)

    seq_map = {}
    for (prog, nb), times in grupos.items():
        if prog == "n-body":
            seq_map[nb] = times

    if not seq_map:
        raise SystemExit(
            f"No se encontraron datos para 'n-body' en {CSV_PATH}. "
            "Necesitamos al menos una entrada secuencial por cada nbodies para calcular SpeedUp."
        )

    prefer_order = ["n-body", "cpu_simd", "build/n-body_gpu"]
    keys = list(grupos.keys())
    def keyfn(item):
        prog, nb = item
        return (prefer_order.index(prog) if prog in prefer_order else 999, prog, nb)

    keys.sort(key=keyfn)

    filas = []
    for prog, nb in keys:
        times = grupos.get((prog, nb), [])
        if not times:
            continue
        seq_times = seq_map.get(nb)
        fila = calcular_fila(prog, times, seq_times=seq_times)
        fila["nbodies"] = nb
        filas.append(fila)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(filas)

    print(f"Resultados guardados en: {OUTPUT_PATH}")
    for f in filas:
        print(
            f"  {f['program']:20s} nb={f['nbodies']:6d}  AM={f['media_aritmetica']:.6f}s  SpeedUp={f['SpeedUp']}"
        )