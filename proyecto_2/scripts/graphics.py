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

required_cols = {"program", "nbodies", "media_aritmetica", "IC", "SpeedUp", "Eficiencia"}
missing = required_cols - set(df.columns)
if missing:
        raise SystemExit(f"Faltan columnas en {CSV_PATH}: {sorted(missing)}")


df["nbodies"] = df["nbodies"].astype(int)
df = df.sort_values(["nbodies"])

programs = df["program"].unique().tolist()
colors = {
        "n-body": "#4CAF50",
        "cpu_simd": "#2196F3",
        "build/n-body_gpu": "#FF9800",
}
bar_colors = [colors.get(p, "#9E9E9E") for p in programs]


pref_order = ["n-body", "cpu_simd", "build/n-body_gpu"]
programs_sorted = [p for p in pref_order if p in programs] + [p for p in programs if p not in pref_order]

# pivot tables indexed by nbodies
tbl_mean = df.pivot(index="nbodies", columns="program", values="media_aritmetica")
tbl_ic = df.pivot(index="nbodies", columns="program", values="IC")

fig, ax = plt.subplots(figsize=(8, 4))
xs = tbl_mean.index.values
for p in programs_sorted:
        if p not in tbl_mean.columns:
                continue
        ys = tbl_mean[p].values
        yerr = tbl_ic[p].values if p in tbl_ic.columns else None
        ax.errorbar(xs, ys, yerr=yerr, label=p, marker="o", capsize=4)

ax.set_xscale('log')
ax.set_xlabel('nbodies')
ax.set_ylabel('Tiempo de ejecución (s) — media')
ax.set_title('Tiempo de ejecución vs nbodies (IC 95%)')
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "tiempo_vs_nbodies.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: tiempo_vs_nbodies.png")


# ── SpeedUp vs nbodies ─────────────────────────────────────────────────────
tbl_sp = df.pivot(index="nbodies", columns="program", values="SpeedUp")
fig, ax = plt.subplots(figsize=(8, 4))
for p in programs_sorted:
        if p not in tbl_sp.columns:
                continue
        ax.plot(tbl_sp.index.values, tbl_sp[p].values, marker="o", label=p)

ax.set_xscale('log')
ax.set_ylabel('SpeedUp (vs n-body)')
ax.set_xlabel('nbodies')
ax.set_title('SpeedUp vs nbodies')
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "speedup_vs_nbodies.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: speedup_vs_nbodies.png")


# ── Eficiencia vs nbodies ───────────────────────────────────────────────────
tbl_eff = df.pivot(index="nbodies", columns="program", values="Eficiencia")
fig, ax = plt.subplots(figsize=(8, 4))
for p in programs_sorted:
        if p not in tbl_eff.columns:
                continue
        ax.plot(tbl_eff.index.values, tbl_eff[p].values, marker="o", label=p)

ax.set_xscale('log')
ax.set_ylabel('Eficiencia (SpeedUp/units)')
ax.set_xlabel('nbodies')
ax.set_title('Eficiencia vs nbodies')
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "eficiencia_vs_nbodies.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: eficiencia_vs_nbodies.png")


print(f"\nGráficas guardadas en: {OUT_DIR}/")