/* Parts of this code are based on Code taken from https://github.com/NoNumberMan/OpenCLTutorial/blob/main/main.cpp*/
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_TARGET_OPENCL_VERSION 200

#include <cassert>
#include <fstream>
#include <iterator>
#include <string>
#include <CL/cl.h>

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
#define NBODIES 10000
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
constexpr int N = NBODIES;
constexpr float DT = 0.002f;
constexpr int STEPS = NSTEPS;
constexpr float G = 1.0f;
constexpr float MASS = 1.0f;
constexpr float SOFTENING = 0.5f;
constexpr bool SAVE_FRAMES = NBSAVEFRAMES != 0;
constexpr bool VERBOSE_LOGS = NBVERBOSE != 0;
static const char kFramesDir[] = "frames/gpu";

constexpr float PI_F = 3.14159265358979323846f;

// Parametros de cada galaxia (identicas)
constexpr float SPHERE_RADIUS = 1.0f;
constexpr float GALAXY_TOTAL_MASS = MASS * static_cast<float>(N / 2 + 99);
constexpr float BLACK_HOLE_MASS_FRACTION = 100.0f / static_cast<float>(N / 2 + 99);
constexpr float DISK_RADIAL_SCALE = SPHERE_RADIUS;
constexpr float DISK_HEIGHT_SCALE = 0.08f;
constexpr float HERNQUIST_R_MIN = 0.02f;
constexpr float VELOCITY_SCALE = 1.0f;

constexpr float SEPARATION = 10.0f * SPHERE_RADIUS;
constexpr float APPROACH_SPEED = 0.7f;
constexpr float TILT_ANGLE_B = PI_F / 6.0f;

// --- Estructura de un cuerpo --------------------------------------------------
struct Body
{
	float x, y, z;
	float vx, vy, vz;
	float ax, ay, az;
	float mass;
};

void compute_accel(Body *bodies);

// --- Inicialización de condiciones iniciales ----------------------------------
static void init_one_disk(Body *bodies,
						  int offset, int n,
						  float cx, float cy, float cz,
						  float bvx, float bvy, float bvz,
						  float tilt_y,
						  unsigned int *seed)
{
	const float eps = std::numeric_limits<float>::epsilon();

	auto rng = [&]() -> float
	{
		*seed = *seed * 1664525u + 1013904223u;
		return static_cast<float>(*seed & 0x7fffffff) / static_cast<float>(0x7fffffff);
	};

	auto rand_uniform = [&](float lo, float hi) -> float
	{
		return lo + (hi - lo) * rng();
	};

	auto hernquist = [&](float r, float r0, float M) -> float
	{
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
		float u = rand_uniform(eps, 1.0f);
		distances[i] = -DISK_RADIAL_SCALE * std::log(1.0f - u);
		float w = hernquist(distances[i], 1.0f, GALAXY_TOTAL_MASS);
		weights[i - 1] = w;
		wsum += w;
	}

	if (wsum <= 0.0f)
	{
		float ms = (GALAXY_TOTAL_MASS - mass_bh) / static_cast<float>(n - 1);
		for (int i = 1; i < n; i++)
			bodies[offset + i].mass = ms;
	}
	else
	{
		for (int i = 1; i < n; i++)
			bodies[offset + i].mass = (weights[i - 1] / wsum) * (GALAXY_TOTAL_MASS - mass_bh);
	}

	std::vector<float> phi(n, 0.0f);

	bodies[offset].x = 0.0f;
	bodies[offset].y = 0.0f;
	bodies[offset].z = 0.0f;
	bodies[offset].ax = 0.0f;
	bodies[offset].ay = 0.0f;
	bodies[offset].az = 0.0f;

	for (int i = 1; i < n; i++)
	{
		float z_decay = std::exp(-0.5f * distances[i] / DISK_RADIAL_SCALE);
		float z_local = rand_uniform(-1.0f, 1.0f) * DISK_HEIGHT_SCALE * z_decay;
		phi[i] = rand_uniform(0.0f, 2.0f * PI_F);

		bodies[offset + i].x = std::cos(phi[i]) * distances[i];
		bodies[offset + i].y = std::sin(phi[i]) * distances[i];
		bodies[offset + i].z = z_local;
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
			if (distances[j] < r)
				mass_enc += bodies[offset + j].mass;

		float r_eff = std::sqrt(r * r + SOFTENING * SOFTENING);
		float v = VELOCITY_SCALE * std::sqrt(G * mass_enc / r_eff);

		bodies[offset + i].vx = -v * std::sin(phi[i]);
		bodies[offset + i].vy = v * std::cos(phi[i]);
		bodies[offset + i].vz = 0.0f;
	}

	{
		float tm = 0.0f, vmx = 0.0f, vmy = 0.0f, vmz = 0.0f;
		for (int i = 0; i < n; i++)
		{
			tm += bodies[offset + i].mass;
			vmx += bodies[offset + i].mass * bodies[offset + i].vx;
			vmy += bodies[offset + i].mass * bodies[offset + i].vy;
			vmz += bodies[offset + i].mass * bodies[offset + i].vz;
		}
		if (tm > 0.0f)
		{
			vmx /= tm;
			vmy /= tm;
			vmz /= tm;
		}
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
			float xi = bodies[offset + i].x, zi = bodies[offset + i].z;
			float vxi = bodies[offset + i].vx, vzi = bodies[offset + i].vz;
			bodies[offset + i].x = xi * ct + zi * st;
			bodies[offset + i].z = -xi * st + zi * ct;
			bodies[offset + i].vx = vxi * ct + vzi * st;
			bodies[offset + i].vz = -vxi * st + vzi * ct;
		}
	}

	for (int i = 0; i < n; i++)
	{
			bodies[offset + i].x += cx;
			bodies[offset + i].y += cy;
			bodies[offset + i].z += cz;
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
				  -APPROACH_SPEED, 0.0f, 0.0f,
				  0.0f,
				  &seedA);

	unsigned int seedB = 67890u;
	init_one_disk(bodies,
				  nA, nB,
				  -SEPARATION * 0.5f, 0.0f, 0.0f,
				  +APPROACH_SPEED, 0.0f, 0.0f,
				  TILT_ANGLE_B,
				  &seedB);
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
	// Use -rf to avoid interactive prompts when directory is write-protected
	snprintf(cmd, sizeof(cmd), "rm -rf %s", kFramesDir);
	(void)system(cmd);
	mkdir(kFramesDir, 0755);
}

// --- Guardado de frame en archivo binario ------------------------------------
void save_frame(const Body *bodies, int step)
{
	if (!SAVE_FRAMES)
		return;

	char filename[256];
	snprintf(filename, sizeof(filename), "%s/frame_%04d.bin", kFramesDir, step);

	int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0)
	{
		fprintf(stderr, "Error abriendo %s\n", filename);
		return;
	}

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

// ------------- OpenCL environment setup ----------------------------
cl_device_id device = nullptr;
cl_context context = nullptr;
cl_command_queue queue = nullptr;
cl_program program = nullptr;
cl_kernel kernel = nullptr;
cl_mem buf_x = nullptr;
cl_mem buf_y = nullptr;
cl_mem buf_z = nullptr;
cl_mem buf_mass = nullptr;
cl_mem buf_ax = nullptr;
cl_mem buf_ay = nullptr;
cl_mem buf_az = nullptr;
size_t buffer_n = 0;
static bool opencl_initialized = false;
static bool kernel_initialized = false;

// Reads the contents of a file and returns it as a string.
static std::string read_file(const char *fileName)
{
	std::ifstream f(fileName, std::ios::in | std::ios::binary);
	if (!f)
	{
		std::cerr << "Error abriendo archivo: " << fileName << std::endl;
		exit(1);
	}

	std::string res((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	return res;
}

void check_result(cl_int error)
{
	if (error != CL_SUCCESS)
	{
		std::cerr << "OpenCL error: " << error << std::endl;
		exit(1);
	}
}

void print_platforminfo(cl_platform_id platform)
{
	printf(" ----- Platform Info ----- \n");
	char platformName[256];
	size_t platformNameLength;
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, 256, platformName, &platformNameLength);
	printf("Platform name: %s\n", platformName);

	char platformVendor[256];
	size_t platformVendorLength;
	clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 256, platformVendor, &platformVendorLength);
	printf("Platform vendor: %s\n", platformVendor);

	char platformVersion[256];
	size_t platformVersionLength;
	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 256, platformVersion, &platformVersionLength);
	printf("Platform version: %s\n", platformVersion);
}

void print_deviceinfo(cl_device_id device)
{
	printf(" ----- Device Info ----- \n");
	char vendorName[256];
	size_t vendorNameLength;
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, 256, vendorName, &vendorNameLength);
	printf("Device vendor: %s\n", vendorName);

	char deviceName[256];
	size_t deviceNameLength;
	clGetDeviceInfo(device, CL_DEVICE_NAME, 256, deviceName, &deviceNameLength);
	printf("Device name: %s\n", deviceName);

	cl_uint maxComputeUnits;
	size_t maxComputeUnitsLength;
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, &maxComputeUnitsLength);
	printf("Device max compute units: %u\n", maxComputeUnits);

	cl_uint maxWorkItemDimention;
	size_t maxWorkItemDimentionLength;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimention, &maxWorkItemDimentionLength);
	printf("Device max work item dimensions: %u\n", maxWorkItemDimention);

	size_t maxWorkItemSizes[3];
	size_t maxWorkItemSizesLength;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, &maxWorkItemSizesLength);
	printf("Device max work item sizes: %zu, %zu, %zu\n", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

	size_t maxWorkGroupSize;
	size_t maxWorkGroupSizeLength;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, &maxWorkGroupSizeLength);
	printf("Device max work group size: %zu\n", maxWorkGroupSize);

#if defined(CL_DEVICE_MAX_NUM_SUB_GROUPS)
	cl_uint maxNumberOfSubGroups;
	size_t maxNumberOfSubGroupsLength;
	clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS, sizeof(cl_uint), &maxNumberOfSubGroups, &maxNumberOfSubGroupsLength);
	printf("Device max number of sub groups: %u\n", maxNumberOfSubGroups);
#elif defined(CL_DEVICE_MAX_NUM_SUB_GROUPS_KHR)
	cl_uint maxNumberOfSubGroups;
	size_t maxNumberOfSubGroupsLength;
	clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS_KHR, sizeof(cl_uint), &maxNumberOfSubGroups, &maxNumberOfSubGroupsLength);
	printf("Device max number of sub groups: %u\n", maxNumberOfSubGroups);
#else
	printf("Device max number of sub groups: N/A\n");
#endif
}

void init_environment()
{
	if (opencl_initialized)
		return;

	if (VERBOSE_LOGS)
		printf(" ------- INITIALIZING ENVIRONMENT ------- \n");

	cl_uint numPlatforms = 4;
	cl_platform_id platforms[numPlatforms];
	unsigned int platformCount;
	// Query for all available OpenCL platforms on the system, and store them in the platforms array.
	// The number of platforms found is stored in platformCount.
	cl_int platformResult = clGetPlatformIDs(numPlatforms, platforms, &platformCount);

	check_result(platformResult);

	if (VERBOSE_LOGS)
	{
		for (unsigned int i = 0; i < platformCount; ++i)
		{
			print_platforminfo(platforms[i]);
		}
	}

	device = nullptr;
	for (unsigned int i = 0; i < platformCount && device == nullptr; ++i)
	{
		cl_device_id devices[4];
		unsigned int deviceCount = 0;
		cl_int deviceResult = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 4, devices, &deviceCount);
		if (deviceResult == CL_DEVICE_NOT_FOUND || deviceCount == 0)
			continue;
		check_result(deviceResult);
		device = devices[0];
	}

	if (device == nullptr)
	{
		std::cerr << "OpenCL error: no se encontro un dispositivo GPU disponible." << std::endl;
		exit(1);
	}

	if (VERBOSE_LOGS)
		print_deviceinfo(device);

	// Create an OpenCL context on the first available platform using the GPU, and create a command queue on that context.
	cl_int contextResult;
	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &contextResult);
	check_result(contextResult);

	cl_int commandQueueResult;
	queue = clCreateCommandQueueWithProperties(context, device, 0, &commandQueueResult);
	check_result(commandQueueResult);

	if (VERBOSE_LOGS)
		printf(" ------- ENVIRONMENT INITIALIZED ------- \n");

	opencl_initialized = true;
}

void cleanup_environment()
{
	if (!opencl_initialized)
		return;

	if (buf_x)
		clReleaseMemObject(buf_x);
	if (buf_y)
		clReleaseMemObject(buf_y);
	if (buf_z)
		clReleaseMemObject(buf_z);
	if (buf_mass)
		clReleaseMemObject(buf_mass);
	if (buf_ax)
		clReleaseMemObject(buf_ax);
	if (buf_ay)
		clReleaseMemObject(buf_ay);
	if (buf_az)
		clReleaseMemObject(buf_az);

	if (kernel)
		clReleaseKernel(kernel);
	if (program)
		clReleaseProgram(program);

	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	buf_x = nullptr;
	buf_y = nullptr;
	buf_z = nullptr;
	buf_mass = nullptr;
	buf_ax = nullptr;
	buf_ay = nullptr;
	buf_az = nullptr;
	program = nullptr;
	kernel = nullptr;
	buffer_n = 0;
	kernel_initialized = false;
	queue = nullptr;
	context = nullptr;
	device = nullptr;
	opencl_initialized = false;
}

void compute_accel(Body *bodies)
{
	init_environment();

	if (!kernel_initialized)
	{
		std::string s = read_file("./models/OpenCL/compute_accel.cl");
		if (s.empty())
		{
			std::cerr << "OpenCL error: kernel vacio o no encontrado." << std::endl;
			exit(1);
		}

		const char *programSource = s.c_str();
		size_t length = s.size();
		cl_int programResult;
		fprintf(stderr, "[DEBUG] Creating program from source...\n"); fflush(stderr);
		program = clCreateProgramWithSource(context, 1, &programSource, &length, &programResult);
		check_result(programResult);

		const char *buildOptions = "-cl-fp32-correctly-rounded-divide-sqrt";
		fprintf(stderr, "[DEBUG] Building program with options '%s'...\n", buildOptions); fflush(stderr);
		cl_int programBuildResult = clBuildProgram(program, 1, &device, buildOptions, nullptr, nullptr);
		if (programBuildResult != CL_SUCCESS)
		{
			size_t logLength = 0;
			cl_int logSizeResult = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLength);
			check_result(logSizeResult);
			std::string log(logLength, '\0');
			cl_int logResult = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLength, &log[0], nullptr);
			check_result(logResult);
			std::cerr << "[BUILD_ERROR] " << log << std::endl;
			exit(1);
		}

		fprintf(stderr, "[DEBUG] Program built successfully. Creating kernel...\n"); fflush(stderr);
		cl_int kernelResult;
		kernel = clCreateKernel(program, "compute_accel", &kernelResult);
		check_result(kernelResult);
		kernel_initialized = true;
	}

	if (buffer_n != static_cast<size_t>(N))
	{
		if (buf_x)
			clReleaseMemObject(buf_x);
		if (buf_y)
			clReleaseMemObject(buf_y);
		if (buf_z)
			clReleaseMemObject(buf_z);
		if (buf_mass)
			clReleaseMemObject(buf_mass);
		if (buf_ax)
			clReleaseMemObject(buf_ax);
		if (buf_ay)
			clReleaseMemObject(buf_ay);
		if (buf_az)
			clReleaseMemObject(buf_az);

		cl_int bufResult;
		buf_x = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), nullptr, &bufResult);
		check_result(bufResult);
		buf_y = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), nullptr, &bufResult);
		check_result(bufResult);
		buf_z = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), nullptr, &bufResult);
		check_result(bufResult);
		buf_mass = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), nullptr, &bufResult);
		check_result(bufResult);
		buf_ax = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), nullptr, &bufResult);
		check_result(bufResult);
		buf_ay = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), nullptr, &bufResult);
		check_result(bufResult);
		buf_az = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), nullptr, &bufResult);
		check_result(bufResult);

		buffer_n = N;
	}

	static std::vector<float> x;
	static std::vector<float> y;
	static std::vector<float> z;
	static std::vector<float> mass;
	static std::vector<float> ax;
	static std::vector<float> ay;
	static std::vector<float> az;

	x.resize(N);
	y.resize(N);
	z.resize(N);
	mass.resize(N);
	ax.resize(N);
	ay.resize(N);
	az.resize(N);

	for (int i = 0; i < N; ++i)
	{
		x[i] = bodies[i].x;
		y[i] = bodies[i].y;
		z[i] = bodies[i].z;
		mass[i] = bodies[i].mass;
	}

	check_result(clEnqueueWriteBuffer(queue, buf_x, CL_TRUE, 0, N * sizeof(float), x.data(), 0, nullptr, nullptr));
	check_result(clEnqueueWriteBuffer(queue, buf_y, CL_TRUE, 0, N * sizeof(float), y.data(), 0, nullptr, nullptr));
	check_result(clEnqueueWriteBuffer(queue, buf_z, CL_TRUE, 0, N * sizeof(float), z.data(), 0, nullptr, nullptr));
	check_result(clEnqueueWriteBuffer(queue, buf_mass, CL_TRUE, 0, N * sizeof(float), mass.data(), 0, nullptr, nullptr));

	int n = N;
	float g = G;
	float softening = SOFTENING;

	check_result(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_x));
	check_result(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_y));
	check_result(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_z));
	check_result(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_mass));
	check_result(clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_ax));
	check_result(clSetKernelArg(kernel, 5, sizeof(cl_mem), &buf_ay));
	check_result(clSetKernelArg(kernel, 6, sizeof(cl_mem), &buf_az));
	check_result(clSetKernelArg(kernel, 7, sizeof(int), &n));
	check_result(clSetKernelArg(kernel, 8, sizeof(float), &g));
	check_result(clSetKernelArg(kernel, 9, sizeof(float), &softening));

	const size_t localWorkSize = 64;
	size_t globalWorkSize = ((static_cast<size_t>(N) + localWorkSize - 1) / localWorkSize) * localWorkSize;
	check_result(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));

	check_result(clEnqueueReadBuffer(queue, buf_ax, CL_TRUE, 0, N * sizeof(float), ax.data(), 0, nullptr, nullptr));
	check_result(clEnqueueReadBuffer(queue, buf_ay, CL_TRUE, 0, N * sizeof(float), ay.data(), 0, nullptr, nullptr));
	check_result(clEnqueueReadBuffer(queue, buf_az, CL_TRUE, 0, N * sizeof(float), az.data(), 0, nullptr, nullptr));

	for (int i = 0; i < N; ++i)
	{
		bodies[i].ax = ax[i];
		bodies[i].ay = ay[i];
		bodies[i].az = az[i];
	}
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

	cleanup_environment();
	delete[] bodies;
	return 0;
}