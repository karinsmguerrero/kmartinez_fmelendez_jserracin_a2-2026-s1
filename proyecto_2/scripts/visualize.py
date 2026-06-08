"""
Visualización 3D N-Body — Choque de galaxias (VisPy / OpenGL / GPU).
Lee los frames binarios generados por nbody_collision.cpp.
Formato de frame: N * 4 floats  →  x, y, z, galaxy_id (0=A, 1=B)

Colores:
    Galaxia A  — gradiente azul/cian
    Galaxia B  — gradiente naranja/amarillo
    El agujero negro de cada galaxia (cuerpo 0 y N/2) se muestra más grande.

Controles:
    Arrastrar ratón  — rotar cámara
    Rueda del ratón  — zoom
    Espacio          — pausar / reanudar
    ← / →            — retroceder / avanzar un frame
    Q / Esc          — salir

Uso:
    python visualize.py [frames/secuencial]
"""

import glob
import sys
import numpy as np

# ── Backend ──────────────────────────────────────────────────────────
from vispy import app
app.use_app("pyqt5")
from vispy import scene

# ── Cargar frames ────────────────────────────────────────────────────
frames_dir = "frames/secuencial"
if len(sys.argv) > 1:
    frames_dir = sys.argv[1].rstrip("/")

frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.bin"))
if not frame_files:
    print(f"Error: No se encontraron archivos en {frames_dir}/.")
    print("Ejecuta primero el modelo correspondiente.")
    sys.exit(1)

print(f"Cargando {len(frame_files)} frames desde {frames_dir}/...")
frames_pos = []   # (num_frames, N, 3)  — solo xyz
frames_gid = None # (N,)               — galaxy_id fijo en frame 0

for idx, f in enumerate(frame_files):
    raw = np.fromfile(f, dtype=np.float32)
    data = raw.reshape(-1, 4)          # (N, 4)
    frames_pos.append(data[:, :3])
    if idx == 0:
        frames_gid = data[:, 3].astype(np.int32)  # 0 o 1, constante

frames_pos = np.array(frames_pos, dtype=np.float32)  # (F, N, 3)
num_frames, n_bodies, _ = frames_pos.shape
print(f"  {num_frames} frames, {n_bodies} cuerpos.")

n_A = int(np.sum(frames_gid == 0))
n_B = int(np.sum(frames_gid == 1))
print(f"  Galaxia A: {n_A} cuerpos  |  Galaxia B: {n_B} cuerpos")

# ── Colores por galaxia ───────────────────────────────────────────────
# Galaxia A: azul oscuro → cian brillante
# Galaxia B: naranja oscuro → amarillo brillante
colors = np.zeros((n_bodies, 4), dtype=np.float32)

mask_A = frames_gid == 0
mask_B = frames_gid == 1

t_A = np.linspace(0.0, 1.0, n_A, dtype=np.float32)
colors[mask_A, 0] = 0.0 + 0.2 * t_A        # R: casi cero
colors[mask_A, 1] = 0.5 + 0.5 * t_A        # G: medio → alto
colors[mask_A, 2] = 1.0                     # B: siempre alto
colors[mask_A, 3] = 0.85                    # A

t_B = np.linspace(0.0, 1.0, n_B, dtype=np.float32)
colors[mask_B, 0] = 1.0                     # R: siempre alto
colors[mask_B, 1] = 0.4 + 0.6 * t_B        # G: medio → alto (naranja→amarillo)
colors[mask_B, 2] = 0.0 + 0.2 * t_B        # B: casi cero
colors[mask_B, 3] = 0.85                    # A

# ── Tamaños: agujeros negros más grandes ─────────────────────────────
sizes = np.full(n_bodies, 3.0, dtype=np.float32)
# El primer cuerpo de cada galaxia es el agujero negro
idx_bh_A = np.where(mask_A)[0][0]
idx_bh_B = np.where(mask_B)[0][0]
sizes[idx_bh_A] = 10.0
sizes[idx_bh_B] = 10.0

# ── Canvas VisPy ──────────────────────────────────────────────────────
canvas = scene.SceneCanvas(
    title=f"Colisión de galaxias — Frame 0001 / {num_frames:04d}",
    bgcolor="black",
    size=(1280, 960),
    keys="interactive",
)
view = canvas.central_widget.add_view()

data_range = float(np.max(np.abs(frames_pos[0])))
cam = scene.cameras.TurntableCamera(
    fov=50,
    distance=data_range * 3.0,
    center=(0, 0, 0),
    elevation=25,
    azimuth=30,
)
view.camera = cam

# ── Scatter GPU ───────────────────────────────────────────────────────
scatter = scene.visuals.Markers(parent=view.scene)
scatter.set_data(
    pos=frames_pos[0],
    face_color=colors,
    size=sizes,
    edge_width=0,
    edge_color=None,
)

# ── Estado ────────────────────────────────────────────────────────────
state = {"idx": 0, "paused": False}

def advance(ev=None, delta=1):
    state["idx"] = (state["idx"] + delta) % num_frames
    idx = state["idx"]
    scatter.set_data(
        pos=frames_pos[idx],
        face_color=colors,
        size=sizes,
        edge_width=0,
        edge_color=None,
    )
    canvas.title = f"Colisión de galaxias — Frame {idx + 1:04d} / {num_frames:04d}"
    canvas.update()

# ── Temporizador (~60 FPS) ────────────────────────────────────────────
timer = app.Timer(interval=1.0 / 60, connect=advance, start=True)

# ── Teclas ────────────────────────────────────────────────────────────
@canvas.events.key_press.connect
def on_key(event):
    if event.key is None:
        return
    key = event.key.name
    if key in ("Q", "Escape"):
        canvas.close()
        app.quit()
    elif key == "Space":
        state["paused"] = not state["paused"]
        if state["paused"]:
            timer.stop()
        else:
            timer.start()
    elif key == "Right":
        advance(delta=1)
    elif key == "Left":
        advance(delta=-1)

# ── Mostrar ───────────────────────────────────────────────────────────
canvas.show()
print("Mostrando animación (GPU / OpenGL)...")
print("  Espacio = pausar/reanudar  |  ←/→ = frame a frame  |  Q/Esc = salir")
print(f"  Backend: {app.use_app().backend_name}")
print(f"  Galaxia A = azul/cian  |  Galaxia B = naranja/amarillo")
app.run()