#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <iostream>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include <assert.h>
// Optimizations
#define STREAM_COMPACTION true
#define SORT_MATERIAL false
#define CACHE_BOUNCE false
#define RAY_CULLING true
// Effects
#define AA true
#define MOTION_BLUR false
#define ERRORCHECK false
#define DENOISE true

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void copy_data(float *dev_data, glm::vec3* dev_vec_data, int offset, glm::ivec2 resolution, float iter) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int size = resolution.x * resolution.y;
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = dev_vec_data[index];
		// Each thread writes one pixel location in the texture (textel)
		dev_data[index + offset] = pix.x / iter;
		dev_data[index + size + offset] = pix.y / iter;
		dev_data[index + size * 2 + offset] = pix.z / iter;
	}
}

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));

	if (CACHE_BOUNCE)
		cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));

	// Denoiser stuff

	cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_depth, pixelcount * sizeof(float));
	cudaMalloc(&dev_tensor, pixelcount * 10 * sizeof(float));

	cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_depth, 0, pixelcount * sizeof(float));
	cudaMemset(dev_tensor, 0, pixelcount * 10 * sizeof(float));

	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	// Mesh
	cudaMalloc(&dev_faces, scene->faces.size() * sizeof(Face));
	cudaMemcpy(dev_faces, scene->faces.data(), scene->faces.size() * sizeof(Face), cudaMemcpyHostToDevice);
	// mesh bounding box
	cudaMalloc((void**)&dev_mesh_box, sizeof(MeshBoundingBox));
	cudaMemcpy(dev_mesh_box, &(scene->mesh_box), sizeof(MeshBoundingBox), cudaMemcpyHostToDevice);
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_intersections_cache);
	// mesh
	cudaFree(dev_faces);
	cudaFree(dev_mesh_box);
	// Denoiser
	cudaFree(dev_normal);
	cudaFree(dev_albedo);
	cudaFree(dev_depth);
	// Vector versions of the same data
	cudaFree(dev_tensor);
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments[index].remainingBounces);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		if (AA)
			// antialiasing by jittering the ray
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng))
				);
		else
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

template<typename T>
void printVector(const T *a1, int n, string name) {
	T *print_a = new T[n];
	std::cout << name.c_str() << endl;
	std::cout << "{" << endl;
	cudaMemcpy(print_a, a1, n * sizeof(T), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		std::cout << "\t" << print_a[i] << endl;
	}
	std::cout << "}" << endl;
	delete[]print_a;
}


// computeIntersections handles generating ray intersections
// Generating new rays is handled in your shader(s).
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, Face * face
	, int face_size
	, MeshBoundingBox * mesh_box
	, glm::vec3 *pixel_normals
	, float *pixel_depth
	, ShadeableIntersection * intersections
	, int iter
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int materialid = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				materialid = geoms[i].materialid;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}
		// Check for mesh collision
		if (face_size && RAY_CULLING && RayAABBintersect(pathSegment.ray, mesh_box[0])) {
			for (int i = 0; i < face_size; i++) {
				t = triangleIntersectionTest(face[i], pathSegment.ray, tmp_intersect, tmp_normal, outside);
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					materialid = face[i].materialid;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
				}
			}
		}
		else if (face_size && !RAY_CULLING) {
			for (int i = 0; i < face_size; i++) {
				t = triangleIntersectionTest(face[i], pathSegment.ray, tmp_intersect, tmp_normal, outside);
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					materialid = face[i].materialid;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
				}
			}
		}
		if (materialid == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = materialid;
			intersections[path_index].surfaceNormal = glm::normalize(normal);
			intersections[path_index].is_inside = !outside;
			intersections[path_index].intersect = intersect_point;
		}
		if (DENOISE && depth == 0 && iter == 1 && intersections[path_index].t >= 0){
			pixel_normals[path_index] = normal;
			pixel_depth[path_index] = intersections[path_index].t;
		}
	}
}

__device__ __forceinline__
glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
	return translationMat * rotationMat * scaleMat;
}
//__global__ void test_copy_kernel(at::Tensor torch_data, glm::vec3 * dev_rgb, glm::vec3 * dev_normal, glm::vec3 * dev_albedo, float * dev_depth, int width) {
//	const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//	const int j = (blockIdx.y * blockDim.y) + threadIdx.y;
//	const int index = j * width + i;
//	torch_data.data()[0][0][i][j] = dev_rgb[index].x;
//	//torch_data[0][1][i][j] = dev_rgb[index].y;
//	//torch_data[0][2][i][j] = dev_rgb[index].z;
//	//torch_data[0][3][i][j] = dev_normal[index].x;
//	//torch_data[0][4][i][j] = dev_normal[index].y;
//	//torch_data[0][5][i][j] = dev_normal[index].z;
//	//torch_data[0][6][i][j] = dev_albedo[index].x;
//	//torch_data[0][7][i][j] = dev_albedo[index].y;
//	//torch_data[0][8][i][j] = dev_albedo[index].z;
//	//torch_data[0][9][i][j] = dev_depth[index];
//}

__global__ void moveGeom(Geom * geoms, int geoms_size, float dt)
{
		// naive parse through global geoms
		int geom_index = blockIdx.x * blockDim.x + threadIdx.x;
		if (geom_index >= geoms_size)
			return;
		Geom & geom = geoms[geom_index];
		if (geom.vel == glm::vec3(0))
			return;
		geom.translation += geom.vel * dt;
		geom.transform = buildTransformationMatrix(geom.translation, geom.rotation, geom.scale);
		geom.inverseTransform = glm::inverse(geom.transform);
		geom.invTranspose = glm::inverseTranspose(geom.transform);
}

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, glm::vec3 *pixel_albedo
	, int depth
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths  && pathSegments[idx].remainingBounces != 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].remainingBounces = 0;
				pathSegments[idx].color = pathSegments[idx].color * materialColor * material.emittance;//glm::clamp(pathSegments[idx].color * materialColor * material.emittance, glm::vec3(0.0), glm::vec3(1.0));
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			else{
				//glm::vec3 intersectionPoint;
				//intersectionPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);//pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
				scatterRay(pathSegments[idx], intersection, material, rng);
				--pathSegments[idx].remainingBounces;
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
		if (DENOISE && depth == 0 && iter == 1 && intersection.t >= 0) {
		pixel_albedo[idx] = pathSegments[idx].color;
		}
	}
	
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct path_termination_test
{
	__host__ __device__
		bool operator()(const PathSegment &path)
	{
		return path.remainingBounces > 0; 
	};
};
struct sort_cmp {
	__host__ __device__
		bool operator()(const ShadeableIntersection &s1, const ShadeableIntersection &s2) {
		return s1.materialId < s2.materialId;
	}
};
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;
	assert(((!CACHE_BOUNCE && AA) || (!AA)));
	assert(((!CACHE_BOUNCE && MOTION_BLUR) || (!MOTION_BLUR)));

	///////////////////////////////////////////////////////////////////////////


	if (MOTION_BLUR && !(iter%4) && iter < 3000) {
		// Move geoms 
		int geom_size = hst_scene->geoms.size();
		dim3 numblocksPathSegmentTracing = (geom_size + blockSize1d - 1) / blockSize1d;
		moveGeom << <numblocksPathSegmentTracing, blockSize1d >> > (dev_geoms, geom_size, 0.10f);
	}

	// perform one iteration of path tracing
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		if (depth == 0 && CACHE_BOUNCE && iter == 1) {
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (depth,
				num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_faces, hst_scene->faces.size(), dev_mesh_box, dev_normal, dev_depth, dev_intersections, iter);
			// cache
			cudaMemcpy(dev_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else if (depth == 0 && CACHE_BOUNCE) {
			// use cache
			cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (depth,
				num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_faces, hst_scene->faces.size(), dev_mesh_box, dev_normal, dev_depth, dev_intersections, iter);
		}
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_albedo,
			depth
			);
		depth++;
		checkCUDAError("Shader"); 
		if (STREAM_COMPACTION) {
			dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, path_termination_test());// split data into completed and non completed
			num_paths = dev_path_end - dev_paths; // update end point to reflect new pivot
		}
		if (SORT_MATERIAL) {
			thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sort_cmp());
;		}
		// Check for termination
		iterationComplete = (num_paths == 0 || depth == traceDepth);
	}
	num_paths = pixelcount;
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather <<<numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////
	/// Try to pass through the network
	
	///////////////////////////////////////////////////////////////////////////
	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
	checkCUDAError("Send to opengl error");
	copy_data<< <blocksPerGrid2d, blockSize2d >> > (dev_tensor, dev_image, 0, cam.resolution, iter);
	if (iter == 1) {
		copy_data << <blocksPerGrid2d, blockSize2d >> > (dev_tensor, dev_normal, pixelcount * 3, cam.resolution, iter); // normals
		copy_data << <blocksPerGrid2d, blockSize2d >> > (dev_tensor, dev_albedo, pixelcount * 6, cam.resolution, iter); // albedos
		cudaMemcpy(dev_tensor + pixelcount * 9, dev_depth, sizeof(float) * pixelcount, cudaMemcpyDeviceToDevice);// depth
	}
	checkCUDAError("Copy error");
	cudaMemcpy(hst_scene->state.host_tensor, dev_tensor, pixelcount * 10 * sizeof(float), cudaMemcpyDeviceToHost);
	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	//if (iter == 1) {
	//	cudaMemcpy(hst_scene->state.normals.data(), dev_normal,
	//		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	//	cudaMemcpy(hst_scene->state.albedos.data(), dev_albedo,
	//		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	//	cudaMemcpy(hst_scene->state.depth.data(), dev_depth,
	//		pixelcount * sizeof(float), cudaMemcpyDeviceToHost);
	//}
	checkCUDAError("pathtrace");
}
