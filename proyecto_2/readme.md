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
    export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocprofiler-systems/lib:$$LD_LIBRARY_PATH
    rocprof-sys-avail --hw-counters --description -c GPU
```

4) Generar un binario con instrumentación
```bash
    rocprof-sys-instrument -M sampling -o profiling/gpu.inst -- build/n-body_gpu
```

5) Ejecutar el perfilado
```bash
    rocprof-sys-run -- profiling/gpu.inst
```
6) Ejecutar sampling
```bash
    rocprof-sys-sample --gpus 0 --G L2CacheHit GL2C_MISS GL2C_MISS_sum GPUBusy GRBM_COUNT LdsLatency MeanOccupancyPerActiveCU MeanOccupancyPerCU OccupancyPercent SQ_WAVE_CYCLES Wavefronts --realtime 0 -- build/n-body_gpu 
```
