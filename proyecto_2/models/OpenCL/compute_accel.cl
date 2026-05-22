__kernel void compute_accel(__global float* a, __global float* b, __global float* c){
	int i = get_global_id(0);
	float sum = a[i] + b[i];
	c[i] = sum;
} 


/*
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
*/