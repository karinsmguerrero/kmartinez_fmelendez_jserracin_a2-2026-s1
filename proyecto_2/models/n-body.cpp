/*
 * Simulación N-Body Gravitacional Secuencial
 * ===========================================
 *
 * Integrador Leapfrog KDK (Kick-Drift-Kick):
 * - Kick₁: v(t + DT/2) = v(t) + a(t) * DT/2
 * - Drift: x(t + DT) = x(t) + v(t+DT/2) * DT
 * - Force: a(t + DT) = F(x(t+DT)) / m
 * - Kick₂: v(t + DT) = v(t+DT/2) + a(t+DT) * DT/2
 *
 * Guardado de frames: Binario float32 [x0,y0,z0,gid0, x1,y1,z1,gid1, ...]
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifndef NBODIES
#define NBODIES 1000
#endif

#ifndef NSTEPS
#define NSTEPS 600
#endif

#ifndef NBSAVEFRAMES
#define NBSAVEFRAMES 1
#endif

#ifndef NBVERBOSE
#define NBVERBOSE 1
#endif

// --- Parámetros de la simulación ----------------------------------------------
constexpr int N            = NBODIES;
constexpr float DT         = 0.002f;
constexpr int STEPS        = NSTEPS;
constexpr float G          = 1.0f;
constexpr float MASS       = 1.0f;
constexpr float SOFTENING  = 0.5f;
constexpr bool SAVE_FRAMES = NBSAVEFRAMES != 0;
constexpr bool VERBOSE_LOGS = NBVERBOSE != 0;
static const char kFramesDir[] = "frames/secuencial";

constexpr float PI_F = 3.14159265358979323846f;


// Parametros de cada galaxia (identicas)
constexpr float SPHERE_RADIUS            = 1.0f;
constexpr float GALAXY_TOTAL_MASS        = MASS * static_cast<float>(N / 2 + 99);
constexpr float BLACK_HOLE_MASS_FRACTION = 100.0f / static_cast<float>(N / 2 + 99);
constexpr float DISK_RADIAL_SCALE        = SPHERE_RADIUS;
constexpr float DISK_HEIGHT_SCALE        = 0.08f;
constexpr float HERNQUIST_R_MIN          = 0.02f;
constexpr float VELOCITY_SCALE           = 1.0f;

constexpr float SEPARATION     = 10.0f * SPHERE_RADIUS;
constexpr float APPROACH_SPEED = 0.7f;
constexpr float TILT_ANGLE_B   = PI_F / 6.0f;

// --- Estructura de un cuerpo --------------------------------------------------
struct Body
{
    float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
    float mass;
};


// --- Inicialización de condiciones iniciales ----------------------------------
static void init_one_disk(Body *bodies,
                          int offset, int n,
                          float cx, float cy, float cz,
                          float bvx, float bvy, float bvz,
                          float tilt_y,
                          unsigned int *seed)
{
    const float eps = std::numeric_limits<float>::epsilon();

    auto rng = [&]() -> float {
        *seed = *seed * 1664525u + 1013904223u;
        return static_cast<float>(*seed & 0x7fffffff) / static_cast<float>(0x7fffffff);
    };

    auto rand_uniform = [&](float lo, float hi) -> float {
        return lo + (hi - lo) * rng();
    };

    auto hernquist = [&](float r, float r0, float M) -> float {
        float rr = std::max(r, HERNQUIST_R_MIN);
        return (M / (2.0f * PI_F)) * (r0 / (rr * std::pow(r0 + rr, 3.0f)));
    };

    const float mass_bh = GALAXY_TOTAL_MASS * BLACK_HOLE_MASS_FRACTION;
    bodies[offset].mass  = mass_bh;

    std::vector<float> distances(n, 0.0f);
    std::vector<float> weights(n - 1, 0.0f);
    float wsum = 0.0f;

    for (int i = 1; i < n; i++)
    {
        float u      = rand_uniform(eps, 1.0f);
        distances[i]  = -DISK_RADIAL_SCALE * std::log(1.0f - u);
        float w      = hernquist(distances[i], 1.0f, GALAXY_TOTAL_MASS);
        weights[i - 1] = w;
        wsum          += w;
    }

    if (wsum <= 0.0f)
    {
        float ms = (GALAXY_TOTAL_MASS - mass_bh) / static_cast<float>(n - 1);
        for (int i = 1; i < n; i++) bodies[offset + i].mass = ms;
    }
    else
    {
        for (int i = 1; i < n; i++)
            bodies[offset + i].mass = (weights[i - 1] / wsum) * (GALAXY_TOTAL_MASS - mass_bh);
    }

    std::vector<float> phi(n, 0.0f);

    bodies[offset].x  = 0.0f; bodies[offset].y  = 0.0f; bodies[offset].z  = 0.0f;
    bodies[offset].ax = 0.0f; bodies[offset].ay = 0.0f; bodies[offset].az = 0.0f;

    for (int i = 1; i < n; i++)
    {
        float z_decay = std::exp(-0.5f * distances[i] / DISK_RADIAL_SCALE);
        float z_local = rand_uniform(-1.0f, 1.0f) * DISK_HEIGHT_SCALE * z_decay;
        phi[i]         = rand_uniform(0.0f, 2.0f * PI_F);

        bodies[offset + i].x  = std::cos(phi[i]) * distances[i];
        bodies[offset + i].y  = std::sin(phi[i]) * distances[i];
        bodies[offset + i].z  = z_local;
        bodies[offset + i].ax = bodies[offset + i].ay = bodies[offset + i].az = 0.0f;
    }

    bodies[offset].vx = bodies[offset].vy = bodies[offset].vz = 0.0f;

    for (int i = 1; i < n; i++)
    {
        float r = distances[i];
        if (r <= eps) { bodies[offset + i].vx = bodies[offset + i].vy = bodies[offset + i].vz = 0.0f; continue; }

        float mass_enc = 0.0f;
        for (int j = 0; j < n; j++)
            if (distances[j] < r) mass_enc += bodies[offset + j].mass;

        float r_eff = std::sqrt(r * r + SOFTENING * SOFTENING);
        float v     = VELOCITY_SCALE * std::sqrt(G * mass_enc / r_eff);

        bodies[offset + i].vx = -v * std::sin(phi[i]);
        bodies[offset + i].vy =  v * std::cos(phi[i]);
        bodies[offset + i].vz = 0.0f;
    }

    {
        float tm = 0.0f, vmx = 0.0f, vmy = 0.0f, vmz = 0.0f;
        for (int i = 0; i < n; i++)
        {
            tm  += bodies[offset + i].mass;
            vmx += bodies[offset + i].mass * bodies[offset + i].vx;
            vmy += bodies[offset + i].mass * bodies[offset + i].vy;
            vmz += bodies[offset + i].mass * bodies[offset + i].vz;
        }
        if (tm > 0.0f) { vmx /= tm; vmy /= tm; vmz /= tm; }
        for (int i = 0; i < n; i++)
        {
            bodies[offset + i].vx -= vmx;
            bodies[offset + i].vy -= vmy;
            bodies[offset + i].vz -= vmz;
        }
    }

    if (std::fabs(tilt_y) > 1e-9f)
    {
        float ct = std::cos(tilt_y);
        float st = std::sin(tilt_y);
        for (int i = 0; i < n; i++)
        {
            float xi  = bodies[offset + i].x,  zi  = bodies[offset + i].z;
            float vxi = bodies[offset + i].vx, vzi = bodies[offset + i].vz;
            bodies[offset + i].x  =  xi * ct + zi * st;
            bodies[offset + i].z  = -xi * st + zi * ct;
            bodies[offset + i].vx =  vxi* ct + vzi* st;
            bodies[offset + i].vz = -vxi* st + vzi* ct;
        }
    }

    for (int i = 0; i < n; i++)
    {
        bodies[offset + i].x  += cx;
        bodies[offset + i].y  += cy;
        bodies[offset + i].z  += cz;
        bodies[offset + i].vx += bvx;
        bodies[offset + i].vy += bvy;
        bodies[offset + i].vz += bvz;
    }
}

void init_bodies(Body *bodies)
{
    int nA = N / 2;
    int nB = N - nA;

    unsigned int seedA = 12345u;
    init_one_disk(bodies,
                  0, nA,
                  +SEPARATION * 0.5f, 0.0f, 0.0f,
                  -APPROACH_SPEED,   0.0f, 0.0f,
                  0.0f,
                  &seedA);

    unsigned int seedB = 67890u;
    init_one_disk(bodies,
                  nA, nB,
                  -SEPARATION * 0.5f, 0.0f, 0.0f,
                  +APPROACH_SPEED,   0.0f, 0.0f,
                  TILT_ANGLE_B,
                  &seedB);
}

// --- Cálculo de aceleraciones gravitacionales (O(N²)) ------------------------
void compute_accel(Body *bodies)
{
    for (int i = 0; i < N; i++)
    {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        for (int j = 0; j < N; j++)
        {
            if (i == j) continue;

            float dx = bodies[j].x - bodies[i].x;

            float dy = bodies[j].y - bodies[i].y;

            float dz = bodies[j].z - bodies[i].z;

            float dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING * SOFTENING;

            float inv_dist = 1.0f / std::sqrt(dist_sq);

            float inv_dist3 = inv_dist * inv_dist * inv_dist;

            float f = G * bodies[j].mass * inv_dist3;

            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }

        bodies[i].ax = ax;
        bodies[i].ay = ay;
        bodies[i].az = az;
    }
}

void integrate_drift(Body *bodies)
{
    for (int i = 0; i < N; i++)
    {
        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;
        bodies[i].z += bodies[i].vz * DT;
    }
}

void integrate_kick(Body *bodies)
{
    for (int i = 0; i < N; i++)
    {
        bodies[i].vx += 0.5f * bodies[i].ax * DT;
        bodies[i].vy += 0.5f * bodies[i].ay * DT;
        bodies[i].vz += 0.5f * bodies[i].az * DT;
    }
}

void integrate(Body *bodies)
{
    integrate_kick(bodies);
    integrate_drift(bodies);
    compute_accel(bodies);
    integrate_kick(bodies);
}

static void prepare_frames_dir()
{
    if (!SAVE_FRAMES)
        return;

    struct stat st;
    if (stat("frames", &st) != 0)
        mkdir("frames", 0755);

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rm -dr %s", kFramesDir);
    (void)system(cmd);
    mkdir(kFramesDir, 0755);
}

// --- Guardado de frame en archivo binario ------------------------------------
void save_frame(const Body *bodies, int step)
{
    if (!SAVE_FRAMES) return;

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/frame_%04d.bin", kFramesDir, step);

    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "Error abriendo %s\n", filename); return; }

    // Mismo formato que collision: 4 floats por cuerpo (x, y, z, galaxy_id)
    float *buf = new float[N * 4];
    int nA = N / 2;
    for (int i = 0; i < N; i++)
    {
        buf[i * 4 + 0] = static_cast<float>(bodies[i].x);
        buf[i * 4 + 1] = static_cast<float>(bodies[i].y);
        buf[i * 4 + 2] = static_cast<float>(bodies[i].z);
        buf[i * 4 + 3] = (i < nA) ? 0.0f : 1.0f;
    }
    write(fd, buf, N * 4 * sizeof(float));
    delete[] buf;
    close(fd);
}


int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    if (SAVE_FRAMES)
        prepare_frames_dir();

    Body *bodies = new Body[N];
    init_bodies(bodies);
    compute_accel(bodies);

    if (VERBOSE_LOGS)
    {
        printf("N-Body Base: %d cuerpos (%d por galaxia), %d pasos (secuencial)\n",
               N, N / 2, STEPS);
        printf("Separacion inicial: %.2f  |  Velocidad de aproximacion: %.2f\n",
               SEPARATION, APPROACH_SPEED);
    }

    for (int step = 0; step < STEPS; step++)
    {
        save_frame(bodies, step + 1);
        integrate(bodies);

        if (VERBOSE_LOGS && ((step + 1) % 100 == 0 || step == 0))
            printf("  Frame %04d/%04d\n", step + 1, STEPS);
    }


    if (VERBOSE_LOGS)
    {
        if (SAVE_FRAMES)
            printf("Frames guardados en %s/\n", kFramesDir);
        printf("Listo.\n");
    }


    delete[] bodies;
    return 0;
}