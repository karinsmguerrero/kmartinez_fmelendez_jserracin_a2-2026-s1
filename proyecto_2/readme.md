Proyecto Grupal 2 de Arquitectura de Computadores

Framework experimental de modelos de ejecución Data-Level Parallelism

Visualizacion
-------------
1) Generar frames: make run o make run-gpu
2) Crear venv e instalar deps: make setup-venv
3) Ejecutar visualizador: make run-visualize

Dependencias Python: requirements.txt

Perfilado de GPU
-------------
1) Instalar ROCm
2) Instalar ROCm Systems Profiler: 
Seguir sección Building Dyninst alongside ROCm Systems Profiler
3) Preparar el ambiente de ejecución
```bash
    echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
	source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh
	export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib
```

4) Generar un binario con instrumentación
```bash
    rocprof-sys-instrument -M sampling -o profiling/gpu.inst -- build/n-body_gpu
```

5) Ejecutar el perfilado
```bash
    rocprof-sys-run -- profiling/gpu.inst
```