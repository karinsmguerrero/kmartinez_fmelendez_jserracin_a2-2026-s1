/* Parts of this code are based on Code taken from https://github.com/NoNumberMan/OpenCLTutorial/blob/main/main.cpp*/
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <cassert>
#include <fstream>
#include "CL/opencl.h"

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
constexpr int N = NBODIES;
constexpr double DT = 0.002;
constexpr int STEPS = NSTEPS;
constexpr double G = 1.0;
constexpr double MASS = 1.0;
constexpr double SOFTENING = 0.5;
constexpr bool SAVE_FRAMES = NBSAVEFRAMES != 0;
constexpr bool VERBOSE_LOGS = NBVERBOSE != 0;

// Parametros de cada galaxia (identicas)
constexpr double SPHERE_RADIUS = 1.0;
constexpr double GALAXY_TOTAL_MASS = MASS * (N / 2 + 99.0);
constexpr double BLACK_HOLE_MASS_FRACTION = 100.0 / (N / 2 + 99.0);
constexpr double DISK_RADIAL_SCALE = SPHERE_RADIUS;
constexpr double DISK_HEIGHT_SCALE = 0.08;
constexpr double HERNQUIST_R_MIN = 0.02;
constexpr double VELOCITY_SCALE = 1.0;

constexpr double SEPARATION = 10.0 * SPHERE_RADIUS;
constexpr double APPROACH_SPEED = 0.7;
constexpr double TILT_ANGLE_B = M_PI / 6.0;

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

	auto rng = [&]() -> double
	{
		*seed = *seed * 1664525u + 1013904223u;
		return (*seed & 0x7fffffff) / static_cast<double>(0x7fffffff);
	};

	auto rand_uniform = [&](double lo, double hi) -> double
	{
		return lo + (hi - lo) * rng();
	};

	auto rand_normal = [&]() -> double
	{
		double u1 = rand_uniform(eps, 1.0);
		double u2 = rand_uniform(eps, 1.0);
		return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
	};

	auto hernquist = [&](double r, double r0, double M) -> double
	{
		double rr = std::max(r, HERNQUIST_R_MIN);
		return (M / (2.0 * M_PI)) * (r0 / (rr * std::pow(r0 + rr, 3.0)));
	};

	const double mass_bh = GALAXY_TOTAL_MASS * BLACK_HOLE_MASS_FRACTION;
	bodies[offset].mass = mass_bh;

	std::vector<double> distances(n, 0.0);
	std::vector<double> weights(n - 1, 0.0);
	double wsum = 0.0;

	for (int i = 1; i < n; i++)
	{
		double u = rand_uniform(eps, 1.0);
		distances[i] = -DISK_RADIAL_SCALE * std::log(1.0 - u);
		double w = hernquist(distances[i], 1.0, GALAXY_TOTAL_MASS);
		weights[i - 1] = w;
		wsum += w;
	}

	if (wsum <= 0.0)
	{
		double ms = (GALAXY_TOTAL_MASS - mass_bh) / (n - 1);
		for (int i = 1; i < n; i++)
			bodies[offset + i].mass = ms;
	}
	else
	{
		for (int i = 1; i < n; i++)
			bodies[offset + i].mass = (weights[i - 1] / wsum) * (GALAXY_TOTAL_MASS - mass_bh);
	}

	std::vector<double> phi(n, 0.0);

	bodies[offset].x = 0.0;
	bodies[offset].y = 0.0;
	bodies[offset].z = 0.0;
	bodies[offset].ax = 0.0;
	bodies[offset].ay = 0.0;
	bodies[offset].az = 0.0;

	for (int i = 1; i < n; i++)
	{
		double z_decay = std::exp(-0.5 * distances[i] / DISK_RADIAL_SCALE);
		double z_local = rand_uniform(-1.0, 1.0) * DISK_HEIGHT_SCALE * z_decay;
		phi[i] = rand_uniform(0.0, 2.0 * M_PI);

		bodies[offset + i].x = std::cos(phi[i]) * distances[i];
		bodies[offset + i].y = std::sin(phi[i]) * distances[i];
		bodies[offset + i].z = z_local;
		bodies[offset + i].ax = bodies[offset + i].ay = bodies[offset + i].az = 0.0;
	}

	bodies[offset].vx = bodies[offset].vy = bodies[offset].vz = 0.0;

	for (int i = 1; i < n; i++)
	{
		double r = distances[i];
		if (r <= eps)
		{
			bodies[offset + i].vx = bodies[offset + i].vy = bodies[offset + i].vz = 0.0;
			continue;
		}

		double mass_enc = 0.0;
		for (int j = 0; j < n; j++)
			if (distances[j] < r)
				mass_enc += bodies[offset + j].mass;

		double r_eff = std::sqrt(r * r + SOFTENING * SOFTENING);
		double v = VELOCITY_SCALE * std::sqrt(G * mass_enc / r_eff);

		bodies[offset + i].vx = -v * std::sin(phi[i]);
		bodies[offset + i].vy = v * std::cos(phi[i]);
		bodies[offset + i].vz = 0.0;
	}

	{
		double tm = 0.0, vmx = 0.0, vmy = 0.0, vmz = 0.0;
		for (int i = 0; i < n; i++)
		{
			tm += bodies[offset + i].mass;
			vmx += bodies[offset + i].mass * bodies[offset + i].vx;
			vmy += bodies[offset + i].mass * bodies[offset + i].vy;
			vmz += bodies[offset + i].mass * bodies[offset + i].vz;
		}
		if (tm > 0.0)
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

	if (std::abs(tilt_y) > 1e-9)
	{
		double ct = std::cos(tilt_y);
		double st = std::sin(tilt_y);
		for (int i = 0; i < n; i++)
		{
			double xi = bodies[offset + i].x, zi = bodies[offset + i].z;
			double vxi = bodies[offset + i].vx, vzi = bodies[offset + i].vz;
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
				  +SEPARATION * 0.5, 0.0, 0.0,
				  -APPROACH_SPEED, 0.0, 0.0,
				  0.0,
				  &seedA);

	unsigned int seedB = 67890u;
	init_one_disk(bodies,
				  nA, nB,
				  -SEPARATION * 0.5, 0.0, 0.0,
				  +APPROACH_SPEED, 0.0, 0.0,
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
	if (!SAVE_FRAMES)
		return;

	char filename[256];
	snprintf(filename, sizeof(filename), "frames/frame_%04d.bin", step);

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

// Reads the contents of a file and returns it as a string.
static std::string read_file(const char *fileName)
{
	std::fstream f;
	f.open(fileName, std::ios_base::in);
	assert(f.is_open());

	std::string res;
	while (!f.eof())
	{
		char c;
		f.get(c);
		res += c;
	}

	f.close();

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

	cl_uint maxNumberOfSubGroups;
	size_t maxNumberOfSubGroupsLength;
	clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS, sizeof(cl_uint), &maxNumberOfSubGroups, &maxNumberOfSubGroupsLength);
	printf("Device max number of sub groups: %u\n", maxNumberOfSubGroups);
}

void init_environment()
{
	printf(" ------- INITIALIZING ENVIRONMENT ------- \n");

	cl_uint numPlatforms = 4;
	cl_platform_id platforms[numPlatforms];
	unsigned int platformCount;
	// Query for all available OpenCL platforms on the system, and store them in the platforms array.
	// The number of platforms found is stored in platformCount.
	cl_int platformResult = clGetPlatformIDs(numPlatforms, platforms, &platformCount);

	check_result(platformResult);

	for (int i = 0; i < platformCount; ++i)
	{
		print_platforminfo(platforms[i]);
	}

	device = nullptr;
	for (int i = 0; i < platformCount && device == nullptr; ++i)
	{
		cl_device_id devices[4];
		unsigned int deviceCount;
		cl_int deviceResult = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 4, devices, &deviceCount);
		check_result(deviceResult);
		// If there is a GPU device, take the first one and stop looking for more platforms.
		device = devices[0];
	}

	print_deviceinfo(device);

	cl_int contextResult;
	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &contextResult);
	check_result(contextResult);

	cl_int commandQueueResult;
	queue = clCreateCommandQueueWithProperties(context, device, 0, &commandQueueResult);
	check_result(commandQueueResult);

	printf(" ------- ENVIRONMENT INITIALIZED ------- \n");
}

void compute_accel(Body *bodies)
{
	// Create an OpenCL context on the first available platform using the GPU,
	// and create a command queue on that context.
	init_environment();

	// Read the kernel file into a string, and create a program from it.
	std::string s = read_file("./models/OpenCL/compute_accel.cl");
	const char *programSource = s.c_str();
	size_t length = 0;
	cl_int programResult;
	cl_program program = clCreateProgramWithSource(context, 1, &programSource, &length, &programResult);
	check_result(programResult);

	// Build the program for the devices, and check for errors.
	cl_int programBuildResult = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
	if (programBuildResult != CL_SUCCESS)
	{
		char log[256];
		size_t logLength;
		cl_int programBuildInfoResult = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 256, log, &logLength);
		check_result(programBuildInfoResult);
		std::cout << log << std::endl;
		assert(log);
	}

	// Create a kernel from the program, and check for errors.
	cl_int kernelResult;
	cl_kernel kernel = clCreateKernel(program, "compute_accel", &kernelResult);
	assert(kernelResult == CL_SUCCESS);

	float vecaData[256];
	float vecbData[256];

	for (int i = 0; i < 256; ++i)
	{
		vecaData[i] = (float)(i * i);
		vecbData[i] = (float)i;
	}

	cl_int vecaResult;
	cl_mem veca = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * sizeof(float), nullptr, &vecaResult);
	assert(vecaResult == CL_SUCCESS);

	cl_int enqueueVecaResult = clEnqueueWriteBuffer(queue, veca, CL_TRUE, 0, 256 * sizeof(float), vecaData, 0, nullptr, nullptr);
	assert(enqueueVecaResult == CL_SUCCESS);

	cl_int vecbResult;
	cl_mem vecb = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * sizeof(float), nullptr, &vecbResult);
	assert(vecbResult == CL_SUCCESS);

	cl_int enqueueVecbResult = clEnqueueWriteBuffer(queue, vecb, CL_TRUE, 0, 256 * sizeof(float), vecbData, 0, nullptr, nullptr);
	assert(enqueueVecbResult == CL_SUCCESS);

	cl_int veccResult;
	cl_mem vecc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 256 * sizeof(float), nullptr, &veccResult);
	assert(veccResult == CL_SUCCESS);

	cl_int kernelArgaResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &veca);
	assert(kernelArgaResult == CL_SUCCESS);
	cl_int kernelArgbResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &vecb);
	assert(kernelArgbResult == CL_SUCCESS);
	cl_int kernelArgcResult = clSetKernelArg(kernel, 2, sizeof(cl_mem), &vecc);
	assert(kernelArgcResult == CL_SUCCESS);

	size_t globalWorkSize = 256;
	size_t localWorkSize = 64;
	size_t workDimensions = 1;
	cl_int enqueueKernelResult = clEnqueueNDRangeKernel(queue, kernel, workDimensions, 0, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
	assert(enqueueKernelResult == CL_SUCCESS);

	float veccData[256];
	cl_int enqueueReadBufferResult = clEnqueueReadBuffer(queue, vecc, CL_TRUE, 0, 256 * sizeof(float), veccData, 0, nullptr, nullptr);
	assert(enqueueReadBufferResult == CL_SUCCESS);

	clFinish(queue);

	std::cout << "Result: \n";
	for (int i = 0; i < 256; ++i)
	{
		std::cout << veccData[i] << std::endl;
	}

	clReleaseMemObject(veca);
	clReleaseMemObject(vecb);
	clReleaseMemObject(vecc);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseDevice(device);
}

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;

	try
	{
		if (SAVE_FRAMES)
			system("rm -dr frames");
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << '\n';
	}

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