# we need to check the type of graphics that we want because since there isnt a variable in each execution like threads for example
# this scipts generates like some really basic graphics

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "resultados.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "graficas"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _order_programs(df: pd.DataFrame) -> pd.DataFrame:
        prefer_order = ["n-body", "cpu_simd", "build/n-body_gpu"]
        df = df.copy()
        df["_order"] = df["program"].apply(lambda p: prefer_order.index(p) if p in prefer_order else 999)
        return df.sort_values(["_order", "program"]).drop(columns=["_order"])


df = pd.read_csv(CSV_PATH)
df = _order_programs(df)

required_cols = {"program", "media_aritmetica", "IC", "SpeedUp", "Eficiencia"}
missing = required_cols - set(df.columns)
if missing:
        raise SystemExit(f"Faltan columnas en {CSV_PATH}: {sorted(missing)}")


programs = df["program"].tolist()
colors = {
        "n-body": "#4CAF50",
        "cpu_simd": "#2196F3",
        "build/n-body_gpu": "#FF9800",
}
bar_colors = [colors.get(p, "#9E9E9E") for p in programs]


# ── Tiempo de ejecución (media) con IC 95% ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(programs, df["media_aritmetica"], color=bar_colors, width=0.55)
ax.errorbar(programs, df["media_aritmetica"], yerr=df["IC"],
                        fmt="none", ecolor="#222222", capsize=6, capthick=1.5)
ax.set_ylabel("Tiempo de ejecución (s) — media")
ax.set_title("Tiempo de ejecución por programa (IC 95%)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT_DIR / "tiempo_media_ic.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: tiempo_media_ic.png")


# ── SpeedUp por programa ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(programs, df["SpeedUp"], color=bar_colors, width=0.55)
ax.set_ylabel("SpeedUp (vs n-body)")
ax.set_title("SpeedUp por programa")
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT_DIR / "speedup_programa.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: speedup_programa.png")


# ── Eficiencia paralela por programa ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(programs, df["Eficiencia"], color=bar_colors, width=0.55)
ax.set_ylabel("Eficiencia (SpeedUp/8)")
ax.set_title("Eficiencia paralela por programa")
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT_DIR / "eficiencia_programa.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: eficiencia_programa.png")


print(f"\nGráficas guardadas en: {OUT_DIR}/")