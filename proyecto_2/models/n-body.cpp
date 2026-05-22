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
constexpr double DT        = 0.002;
constexpr int STEPS        = NSTEPS;
constexpr double G         = 1.0;
constexpr double MASS      = 1.0;
constexpr double SOFTENING = 0.5;
constexpr bool SAVE_FRAMES = NBSAVEFRAMES != 0;
constexpr bool VERBOSE_LOGS = NBVERBOSE != 0;


// Parametros de cada galaxia (identicas)
constexpr double SPHERE_RADIUS            = 1.0;
constexpr double GALAXY_TOTAL_MASS        = MASS * (N / 2 + 99.0);
constexpr double BLACK_HOLE_MASS_FRACTION = 100.0 / (N / 2 + 99.0);
constexpr double DISK_RADIAL_SCALE        = SPHERE_RADIUS;
constexpr double DISK_HEIGHT_SCALE        = 0.08;
constexpr double HERNQUIST_R_MIN          = 0.02;
constexpr double VELOCITY_SCALE           = 1.0;

constexpr double SEPARATION     = 10.0 * SPHERE_RADIUS;
constexpr double APPROACH_SPEED = 0.7;
constexpr double TILT_ANGLE_B   = M_PI / 6.0;

// --- Estructura de un cuerpo --------------------------------------------------
struct Body
{
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
};


// --- Inicialización de condiciones iniciales ----------------------------------
static void init_one_disk(Body *bodies,
                          int offset, int n,
                          double cx, double cy, double cz,
                          double bvx, double bvy, double bvz,
                          double tilt_y,
                          unsigned int *seed)
{
    const double eps = std::numeric_limits<float>::epsilon();

    auto rng = [&]() -> double {
        *seed = *seed * 1664525u + 1013904223u;
        return (*seed & 0x7fffffff) / static_cast<double>(0x7fffffff);
    };

    auto rand_uniform = [&](double lo, double hi) -> double {
        return lo + (hi - lo) * rng();
    };

    auto rand_normal = [&]() -> double {
        double u1 = rand_uniform(eps, 1.0);
        double u2 = rand_uniform(eps, 1.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    auto hernquist = [&](double r, double r0, double M) -> double {
        double rr = std::max(r, HERNQUIST_R_MIN);
        return (M / (2.0 * M_PI)) * (r0 / (rr * std::pow(r0 + rr, 3.0)));
    };

    const double mass_bh = GALAXY_TOTAL_MASS * BLACK_HOLE_MASS_FRACTION;
    bodies[offset].mass  = mass_bh;

    std::vector<double> distances(n, 0.0);
    std::vector<double> weights(n - 1, 0.0);
    double wsum = 0.0;

    for (int i = 1; i < n; i++)
    {
        double u      = rand_uniform(eps, 1.0);
        distances[i]  = -DISK_RADIAL_SCALE * std::log(1.0 - u);
        double w      = hernquist(distances[i], 1.0, GALAXY_TOTAL_MASS);
        weights[i - 1] = w;
        wsum          += w;
    }

    if (wsum <= 0.0)
    {
        double ms = (GALAXY_TOTAL_MASS - mass_bh) / (n - 1);
        for (int i = 1; i < n; i++) bodies[offset + i].mass = ms;
    }
    else
    {
        for (int i = 1; i < n; i++)
            bodies[offset + i].mass = (weights[i - 1] / wsum) * (GALAXY_TOTAL_MASS - mass_bh);
    }

    std::vector<double> phi(n, 0.0);

    bodies[offset].x  = 0.0; bodies[offset].y  = 0.0; bodies[offset].z  = 0.0;
    bodies[offset].ax = 0.0; bodies[offset].ay = 0.0; bodies[offset].az = 0.0;

    for (int i = 1; i < n; i++)
    {
        double z_decay = std::exp(-0.5 * distances[i] / DISK_RADIAL_SCALE);
        double z_local = rand_uniform(-1.0, 1.0) * DISK_HEIGHT_SCALE * z_decay;
        phi[i]         = rand_uniform(0.0, 2.0 * M_PI);

        bodies[offset + i].x  = std::cos(phi[i]) * distances[i];
        bodies[offset + i].y  = std::sin(phi[i]) * distances[i];
        bodies[offset + i].z  = z_local;
        bodies[offset + i].ax = bodies[offset + i].ay = bodies[offset + i].az = 0.0;
    }

    bodies[offset].vx = bodies[offset].vy = bodies[offset].vz = 0.0;

    for (int i = 1; i < n; i++)
    {
        double r = distances[i];
        if (r <= eps) { bodies[offset + i].vx = bodies[offset + i].vy = bodies[offset + i].vz = 0.0; continue; }

        double mass_enc = 0.0;
        for (int j = 0; j < n; j++)
            if (distances[j] < r) mass_enc += bodies[offset + j].mass;

        double r_eff = std::sqrt(r * r + SOFTENING * SOFTENING);
        double v     = VELOCITY_SCALE * std::sqrt(G * mass_enc / r_eff);

        bodies[offset + i].vx = -v * std::sin(phi[i]);
        bodies[offset + i].vy =  v * std::cos(phi[i]);
        bodies[offset + i].vz = 0.0;
    }

    {
        double tm = 0.0, vmx = 0.0, vmy = 0.0, vmz = 0.0;
        for (int i = 0; i < n; i++)
        {
            tm  += bodies[offset + i].mass;
            vmx += bodies[offset + i].mass * bodies[offset + i].vx;
            vmy += bodies[offset + i].mass * bodies[offset + i].vy;
            vmz += bodies[offset + i].mass * bodies[offset + i].vz;
        }
        if (tm > 0.0) { vmx /= tm; vmy /= tm; vmz /= tm; }
        for (int i = 0; i < n; i++)
        {
            bodies[offset + i].vx -= vmx;
            bodies[offset + i].vy -= vmy;
            bodies[offset + i].vz -= vmz;
        }
    }

    if (std::abs(tilt_y) > 1e-9)
    {
        double ct = std::cos(tilt_y);
        double st = std::sin(tilt_y);
        for (int i = 0; i < n; i++)
        {
            double xi  = bodies[offset + i].x,  zi  = bodies[offset + i].z;
            double vxi = bodies[offset + i].vx, vzi = bodies[offset + i].vz;
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
                  +SEPARATION * 0.5, 0.0, 0.0,
                  -APPROACH_SPEED,   0.0, 0.0,
                  0.0,
                  &seedA);

    unsigned int seedB = 67890u;
    init_one_disk(bodies,
                  nA, nB,
                  -SEPARATION * 0.5, 0.0, 0.0,
                  +APPROACH_SPEED,   0.0, 0.0,
                  TILT_ANGLE_B,
                  &seedB);
}

// --- Cálculo de aceleraciones gravitacionales (O(N²)) ------------------------
void compute_accel(Body *bodies)
{
    for (int i = 0; i < N; i++)
    {
        double ax = 0.0, ay = 0.0, az = 0.0;

        for (int j = 0; j < N; j++)
        {
            if (i == j) continue;

            double dx = bodies[j].x - bodies[i].x;

            double dy = bodies[j].y - bodies[i].y;

            double dz = bodies[j].z - bodies[i].z;

            double dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;

            double inv_dist = 1.0 / sqrt(dist_sq);

            double inv_dist3 = inv_dist * inv_dist * inv_dist;

            double f = G * bodies[j].mass * inv_dist3;

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
        bodies[i].vx += 0.5 * bodies[i].ax * DT;
        bodies[i].vy += 0.5 * bodies[i].ay * DT;
        bodies[i].vz += 0.5 * bodies[i].az * DT;
    }
}

void integrate(Body *bodies)
{
    integrate_kick(bodies);
    integrate_drift(bodies);
    compute_accel(bodies);
    integrate_kick(bodies);
}

// --- Guardado de frame en archivo binario ------------------------------------
void save_frame(const Body *bodies, int step)
{
    if (!SAVE_FRAMES) return;

    char filename[256];
    snprintf(filename, sizeof(filename), "frames/frame_%04d.bin", step);

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

    try { if (SAVE_FRAMES) system("rm -dr frames"); }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; }

    if (SAVE_FRAMES)
        mkdir("frames", 0755);

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
            printf("Frames guardados en frames/\n");
        printf("Listo.\n");
    }


    delete[] bodies;
    return 0;
}