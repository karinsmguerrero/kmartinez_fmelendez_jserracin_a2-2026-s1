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
#include <immintrin.h>
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
constexpr int   N            = NBODIES;
constexpr float DT           = 0.002f;
constexpr int   STEPS        = NSTEPS;
constexpr float G            = 1.0f;
constexpr float MASS         = 1.0f;
constexpr float SOFTENING    = 0.5f;
constexpr bool  SAVE_FRAMES  = NBSAVEFRAMES != 0;
constexpr bool  VERBOSE_LOGS = NBVERBOSE != 0;
static const char kFramesDir[] = "frames/simd";

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

 // alineación a 32 bytes (256 bits = ancho de registro AVX2)
 // se usa struct of arrays para poder cargar los 8 valores a la vez
 // deben de estar contiguos para poder vectorizarlos
struct BodySoA
{
    float *x,  *y,  *z;
    float *vx, *vy, *vz;
    float *ax, *ay, *az;
    float *mass;
    int    n;

    explicit BodySoA(int n_) : n(n_)
    {
        auto alloc = [&](float *&ptr) {
            if (posix_memalign(reinterpret_cast<void **>(&ptr), 32,
                               n_ * sizeof(float)) != 0)
            {
                std::cerr << "Error: posix_memalign falló\n";
                std::exit(1);
            }
            std::memset(ptr, 0, n_ * sizeof(float));
        };
        alloc(x);    alloc(y);    alloc(z);
        alloc(vx);   alloc(vy);   alloc(vz);
        alloc(ax);   alloc(ay);   alloc(az);
        alloc(mass);
    }

    ~BodySoA()
    {
        free(x);  free(y);  free(z);
        free(vx); free(vy); free(vz);
        free(ax); free(ay); free(az);
        free(mass);
    }

    BodySoA(const BodySoA &) = delete;
    BodySoA &operator=(const BodySoA &) = delete;
};

// --- Estructura de un cuerpo --------------------------------------------------
struct Body
{
    float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
    float mass;
};

// Conversión AoS → SoA (se hace una sola vez al inicio)
static void aos_to_soa(const Body *aos, BodySoA &soa, int n)
{
    for (int i = 0; i < n; i++)
    {
        soa.x[i]    = aos[i].x;    soa.y[i]  = aos[i].y;    soa.z[i]  = aos[i].z;
        soa.vx[i]   = aos[i].vx;   soa.vy[i] = aos[i].vy;   soa.vz[i] = aos[i].vz;
        soa.ax[i]   = aos[i].ax;   soa.ay[i] = aos[i].ay;   soa.az[i] = aos[i].az;
        soa.mass[i] = aos[i].mass;
    }
}

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
    bodies[offset].mass = mass_bh;

    std::vector<float> distances(n, 0.0f);
    std::vector<float> weights(n - 1, 0.0f);
    float wsum = 0.0f;

    for (int i = 1; i < n; i++)
    {
        float u      = rand_uniform(eps, 1.0f);
        distances[i] = -DISK_RADIAL_SCALE * std::log(1.0f - u);
        float w      = hernquist(distances[i], 1.0f, GALAXY_TOTAL_MASS);
        weights[i-1] = w;
        wsum        += w;
    }

    if (wsum <= 0.0f)
    {
        float ms = (GALAXY_TOTAL_MASS - mass_bh) / static_cast<float>(n - 1);
        for (int i = 1; i < n; i++) bodies[offset + i].mass = ms;
    }
    else
    {
        for (int i = 1; i < n; i++)
            bodies[offset + i].mass = (weights[i-1] / wsum) * (GALAXY_TOTAL_MASS - mass_bh);
    }

    std::vector<float> phi(n, 0.0f);

    bodies[offset].x  = 0.0f; bodies[offset].y  = 0.0f; bodies[offset].z  = 0.0f;
    bodies[offset].ax = 0.0f; bodies[offset].ay = 0.0f; bodies[offset].az = 0.0f;

    for (int i = 1; i < n; i++)
    {
        float z_decay = std::exp(-0.5f * distances[i] / DISK_RADIAL_SCALE);
        float z_local = rand_uniform(-1.0f, 1.0f) * DISK_HEIGHT_SCALE * z_decay;
        phi[i]        = rand_uniform(0.0f, 2.0f * PI_F);

        bodies[offset + i].x  = std::cos(phi[i]) * distances[i];
        bodies[offset + i].y  = std::sin(phi[i]) * distances[i];
        bodies[offset + i].z  = z_local;
        bodies[offset + i].ax = bodies[offset + i].ay = bodies[offset + i].az = 0.0f;
    }

    bodies[offset].vx = bodies[offset].vy = bodies[offset].vz = 0.0f;

    for (int i = 1; i < n; i++)
    {
        float r = distances[i];
        if (r <= eps)
        {
            bodies[offset + i].vx = bodies[offset + i].vy = bodies[offset + i].vz = 0.0f;
            continue;
        }

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
            bodies[offset + i].vx =  vxi * ct + vzi * st;
            bodies[offset + i].vz = -vxi * st + vzi * ct;
        }
    }

    for (int i = 0; i < n; i++)
    {
        bodies[offset + i].x  += cx;  bodies[offset + i].y  += cy;  bodies[offset + i].z  += cz;
        bodies[offset + i].vx += bvx; bodies[offset + i].vy += bvy; bodies[offset + i].vz += bvz;
    }
}

void init_bodies(Body *bodies)
{
    int nA = N / 2;
    int nB = N - nA;

    unsigned int seedA = 12345u;
    init_one_disk(bodies, 0, nA,
                  +SEPARATION * 0.5f, 0.0f, 0.0f,
                  -APPROACH_SPEED, 0.0f, 0.0f,
                  0.0f, &seedA);

    unsigned int seedB = 67890u;
    init_one_disk(bodies, nA, nB,
                  -SEPARATION * 0.5f, 0.0f, 0.0f,
                  +APPROACH_SPEED, 0.0f, 0.0f,
                  TILT_ANGLE_B, &seedB);
}


static inline float hsum_avx(__m256 v)
{
    __m128 lo    = _mm256_castps256_ps128(v);
    __m128 hi    = _mm256_extractf128_ps(v, 1);
    __m128 sum   = _mm_add_ps(lo, hi);
    __m128 shuf  = _mm_movehdup_ps(sum);
    __m128 sum2  = _mm_add_ps(sum, shuf);
    __m128 upper = _mm_movehl_ps(sum2, sum2);
    __m128 res   = _mm_add_ss(sum2, upper);
    return _mm_cvtss_f32(res);
}

void compute_accel_simd(BodySoA &b)
{
    const float soft2 = SOFTENING * SOFTENING;

    const __m256 v_soft2 = _mm256_set1_ps(soft2);
    const __m256 v_G     = _mm256_set1_ps(G);
    const __m256 v_one   = _mm256_set1_ps(1.0f);

    // Múltiplo de 8 más cercano hacia abajo
    const int N8 = N & ~7;

    for (int i = 0; i < N; i++)
    {
        const __m256 xi  = _mm256_set1_ps(b.x[i]);
        const __m256 yi  = _mm256_set1_ps(b.y[i]);
        const __m256 zi  = _mm256_set1_ps(b.z[i]);
        const __m256 v_i = _mm256_set1_ps(static_cast<float>(i));

        __m256 ax_acc = _mm256_setzero_ps();
        __m256 ay_acc = _mm256_setzero_ps();
        __m256 az_acc = _mm256_setzero_ps();

        for (int j = 0; j < N8; j += 8)
        {
            __m256 xj    = _mm256_load_ps(&b.x[j]);
            __m256 yj    = _mm256_load_ps(&b.y[j]);
            __m256 zj    = _mm256_load_ps(&b.z[j]);
            __m256 massj = _mm256_load_ps(&b.mass[j]);

            __m256 dx = _mm256_sub_ps(xj, xi);
            __m256 dy = _mm256_sub_ps(yj, yi);
            __m256 dz = _mm256_sub_ps(zj, zi);

            __m256 dist_sq = _mm256_fmadd_ps(dx, dx, v_soft2);
            dist_sq = _mm256_fmadd_ps(dy, dy, dist_sq);
            dist_sq = _mm256_fmadd_ps(dz, dz, dist_sq);

            __m256 inv_dist  = _mm256_div_ps(v_one, _mm256_sqrt_ps(dist_sq));

            __m256 inv_dist2 = _mm256_mul_ps(inv_dist, inv_dist);
            __m256 inv_dist3 = _mm256_mul_ps(inv_dist2, inv_dist);

            __m256 f = _mm256_mul_ps(v_G, _mm256_mul_ps(massj, inv_dist3));


            //verificacion de if i==j 
            //se hace un mask creando un vector para realizar la comparación
            //en vez de hacer continue lo que hace es poner su aporte en 0
            __m256 v_j = _mm256_set_ps(
                static_cast<float>(j + 7), static_cast<float>(j + 6),
                static_cast<float>(j + 5), static_cast<float>(j + 4),
                static_cast<float>(j + 3), static_cast<float>(j + 2),
                static_cast<float>(j + 1), static_cast<float>(j + 0));

            __m256 mask = _mm256_cmp_ps(v_j, v_i, _CMP_EQ_OQ);
            f = _mm256_andnot_ps(mask, f);

            ax_acc = _mm256_fmadd_ps(f, dx, ax_acc);
            ay_acc = _mm256_fmadd_ps(f, dy, ay_acc);
            az_acc = _mm256_fmadd_ps(f, dz, az_acc);
        }

        float ax = hsum_avx(ax_acc);
        float ay = hsum_avx(ay_acc);
        float az = hsum_avx(az_acc);

        for (int j = N8; j < N; j++)
        {
            if (j == i) continue;

            float dx       = b.x[j] - b.x[i];
            float dy       = b.y[j] - b.y[i];
            float dz       = b.z[j] - b.z[i];
            float dist_sq  = dx*dx + dy*dy + dz*dz + soft2;
            float inv_dist = 1.0f / std::sqrt(dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            float f        = G * b.mass[j] * inv_dist3;

            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }

        b.ax[i] = ax;
        b.ay[i] = ay;
        b.az[i] = az;
    }
}

void integrate_drift_simd(BodySoA &bodies)
{
    for (int i = 0; i < N; i++)
    {
        bodies.x[i] += bodies.vx[i] * DT;
        bodies.y[i] += bodies.vy[i] * DT;
        bodies.z[i] += bodies.vz[i] * DT;
    }
}

void integrate_kick_simd(BodySoA &bodies)
{
    const float half_dt = 0.5f * DT;
    for (int i = 0; i < N; i++)
    {
        bodies.vx[i] += half_dt * bodies.ax[i];
        bodies.vy[i] += half_dt * bodies.ay[i];
        bodies.vz[i] += half_dt * bodies.az[i];
    }
}

void integrate_simd(BodySoA &bodies)
{
    integrate_kick_simd(bodies);
    integrate_drift_simd(bodies);
    compute_accel_simd(bodies);
    integrate_kick_simd(bodies);
}

static void prepare_frames_dir()
{
    if (!SAVE_FRAMES) return;

    struct stat st;
    if (stat("frames", &st) != 0)
        mkdir("frames", 0755);

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rm -dr %s", kFramesDir);
    (void)system(cmd);
    mkdir(kFramesDir, 0755);
}

// --- Guardado de frame en archivo binario ------------------------------------
void save_frame(const BodySoA &bodies, int step)
{
    if (!SAVE_FRAMES) return;

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/frame_%04d.bin", kFramesDir, step);

    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "Error abriendo %s\n", filename); return; }

    float *buf = new float[N * 4];
    int nA = N / 2;
    for (int i = 0; i < N; i++)
    {
        buf[i * 4 + 0] = bodies.x[i];
        buf[i * 4 + 1] = bodies.y[i];
        buf[i * 4 + 2] = bodies.z[i];
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

    Body *bodies_aos = new Body[N];
    init_bodies(bodies_aos);

    // 2. Convertir AoS → SoA para operaciones SIMD eficientes
    BodySoA b(N);
    aos_to_soa(bodies_aos, b, N);
    delete[] bodies_aos;

    // 3. Aceleraciones iniciales en SIMD
    compute_accel_simd(b);

    if (VERBOSE_LOGS)
    {
        printf("N-Body SIMD AVX2: %d cuerpos (%d por galaxia), %d pasos\n",
               N, N / 2, STEPS);
        printf("Separacion inicial: %.2f  |  Velocidad de aproximacion: %.2f\n",
               SEPARATION, APPROACH_SPEED);
        printf("Layout: SoA, precision: float32, vectorizacion: AVX2 (8 floats/ciclo)\n");
    }

    for (int step = 0; step < STEPS; step++)
    {
        save_frame(b, step + 1);
        integrate_simd(b);

        if (VERBOSE_LOGS && ((step + 1) % 100 == 0 || step == 0))
            printf("  Frame %04d/%04d\n", step + 1, STEPS);
    }

    if (VERBOSE_LOGS)
    {
        if (SAVE_FRAMES)
            printf("Frames guardados en %s/\n", kFramesDir);
        printf("Listo.\n");
    }

    return 0;
}