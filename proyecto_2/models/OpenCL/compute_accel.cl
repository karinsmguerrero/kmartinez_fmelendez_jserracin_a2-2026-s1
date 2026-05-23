#pragma OPENCL FP_CONTRACT OFF

__kernel void compute_accel(__global const float *x,
                            __global const float *y,
                            __global const float *z,
                            __global const float *mass,
                            __global float *ax,
                            __global float *ay,
                            __global float *az,
                            int n,
                            float g,
                            float softening)
{
    int i = get_global_id(0);
    if (i >= n)
        return;

    float xi = x[i];
    float yi = y[i];
    float zi = z[i];

    float axi = 0.0f;
    float ayi = 0.0f;
    float azi = 0.0f;
    float soft2 = softening * softening;

    for (int j = 0; j < n; ++j)
    {
        if (i == j)
            continue;

        float dx = x[j] - xi;
        float dy = y[j] - yi;
        float dz = z[j] - zi;

        float dist_sq = dx * dx + dy * dy + dz * dz + soft2;
        float inv_dist = 1.0f / sqrt(dist_sq);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        float f = g * mass[j] * inv_dist3;

        axi += f * dx;
        ayi += f * dy;
        azi += f * dz;
    }

    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}
