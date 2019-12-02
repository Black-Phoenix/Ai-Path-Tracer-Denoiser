#include "main.h"
#include "preview.h"
#include <cstring>
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include <Windows.h>
#include <torch/script.h>
#include <cuda.h>
__global__ void test() {
	int x = 10;
}
static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;
static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;
int movement_sign = 1;
float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
string filename;
int iteration;
int frame_number = 0;
int width;
int height;
#define SAVE_DENOISE false
#define GROUND_TRUTH false
#define DENOISE_RENDER true

//-------------------------------
//-------------MAIN--------------
//-------------------------------
int main(int argc, char** argv) {
	
    startTimeString = currentTimeString();
    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];
	filename = argv[1];
	// Load scene file
	scene = new Scene(sceneFile);
	//load_obj();
    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;
    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);
    cameraPosition = cam.position;
    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}

void view_tensor(at::Tensor t) {
	at::Tensor output;
	output = t.to(at::kCPU);
	

	//std::cout << output.size(0) << "x" << output.size(1) << "x" <<
	//	output.size(2) << "x" << output.size(3) << "Total:" << output.numel() << endl;
	cv::Mat cv_mat = cv::Mat::eye(width, height, CV_32FC3);
	std::memcpy(cv_mat.data, output.data_ptr(), sizeof(float)*output.numel());
	cv::cvtColor(cv_mat, cv_mat, cv::COLOR_RGB2BGR);
	cv::imshow("Output", cv_mat);
}
//void network_prediction_slow(int iter) {
//	cv::Mat img_normals(width, height, CV_32FC3);
//	cv::Mat img_depth(width, height, CV_32FC1);
//	cv::Mat img_albedo(width, height, CV_32FC3);
//	cv::Mat img_rgb(width, height, CV_32FC3);
//	cv::Mat frame;
//	for (int x = 0; x < width; x++) {
//		for (int y = 0; y < height; y++) {
//			int index = x + (y * width);
//			// Normals
//			glm::vec3 dev_normal = renderState->normals[index];
//			cv::Vec3f &normals = img_normals.at<cv::Vec3f>(x, y);
//			normals[0] = dev_normal.x;
//			normals[1] = dev_normal.y;
//			normals[2] = dev_normal.z;
//			// Depth
//			float dev_depth = renderState->depth[index];
//			float &depth = img_depth.at<float>(x, y);
//			depth = dev_depth;
//			// Albedo
//			glm::vec3 dev_albedo = renderState->albedos[index];
//			cv::Vec3f &albedos = img_albedo.at<cv::Vec3f>(x, y);
//			albedos[0] = dev_albedo.x;
//			albedos[1] = dev_albedo.y;
//			albedos[2] = dev_albedo.z;
//			// RGB
//			glm::vec3 dev_rgb = renderState->image[index];
//			cv::Vec3f &rgb = img_rgb.at<cv::Vec3f>(x, y);
//			rgb[0] = dev_rgb.x / iter;
//			rgb[1] = dev_rgb.y / iter;
//			rgb[2] = dev_rgb.z / iter;
//		}
//	}
//	try {
//		vector<cv::Mat> channels;
//		channels.push_back(img_rgb);
//		channels.push_back(img_normals);
//		channels.push_back(img_albedo);
//		channels.push_back(img_depth);
//		for (int i = 0; i < 4; i++) {
//			cv::rotate(channels[i], channels[i], cv::ROTATE_90_CLOCKWISE);
//		}
//		cv::merge(channels, frame);
//		auto input_tensor = torch::from_blob(frame.data, { 1, width, height, 10 });
//		torch::jit::script::Module module;
//		// Deserialize the ScriptModule from a file using torch::jit::load().
//		module = torch::jit::load(R"(C:\Users\Raven\Documents\565_Assignments\Final_Project\Project3-CUDA-Path-Tracer\cpp_autoencoder_780.pt)");
//		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
//		input_tensor = input_tensor.to(at::kCUDA);
//		//// Create a vector of inputs.
//		std::vector<torch::jit::IValue> inputs;
//		inputs.push_back(input_tensor);
//
//		// Execute the model and turn its output into a tensor.
//		at::Tensor output = module.forward(inputs).toTensor();
//		output = output.permute({ 0, 2, 3, 1 });
//		output = output.to(at::kCPU);
//		cout << output.size(0) << "x" << output.size(1) << "x" <<
//			output.size(2) << "x" << output.size(3) << "Total:" << output.numel() << endl;
//		cv::Mat cv_mat = cv::Mat::eye(width, height, CV_32FC3);
//		std::memcpy(cv_mat.data, output.data_ptr(), sizeof(float)*output.numel());
//		cv::cvtColor(cv_mat, cv_mat, cv::COLOR_RGB2BGR);
//		cv::imshow("Output", cv_mat);
//		cv::waitKey(0);
//	}
//	catch (const exception e) {
//		std::cerr << "error\n" << e.what();
//	}
//}
void network_prediction_faster_version(float *rgb) {
	try {
		// Create the tensor
		auto input_tensor = torch::from_blob(rgb, { 1, 10, width, height });
		input_tensor = input_tensor.to(at::kCUDA);
		// Load the model
		static torch::jit::script::Module module = torch::jit::load
		(R"(C:\Users\Raven\Documents\565_Assignments\Final_Project\Project3-CUDA-Path-Tracer\cpp_autoencoder_24.pt)");
		// Create the input stuff
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(input_tensor);
		at::Tensor output = module.forward(inputs).toTensor();
		output = output.permute({ 0, 2, 3, 1 });
		view_tensor(output);
	}
	catch (const exception e) {
		std::cerr << "error\n" << e.what();
	}
	//auto input_tensor = torch::from_blob(rgb, { 1, 3, width, height });
	//input_tensor = input_tensor.permute({ 0, 2, 3, 1 });
	//input_tensor = input_tensor.to(at::kCUDA);
	//view_tensor(input_tensor);
}

int runCuda() {
	
    if (camchanged) {
		iteration = 0;
		Camera &cam = renderState->camera;
	
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);
		if (cameraPosition.y < -5 && movement_sign < 0)
			movement_sign = 1;
		else if (cameraPosition.y > 5 && movement_sign > 0)
			movement_sign = -1;
		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
     }
	if (frame_number > 162) {
		cout << "Done" << endl;
		exit(0);
	}
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }
    if (iteration < renderState->iterations) { // actually render more
        uchar4 *pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
	// Moving camera
	if (!GROUND_TRUTH || (GROUND_TRUTH && iteration >= renderState->iterations)) {
		if (DENOISE_RENDER) {
			network_prediction_faster_version(renderState->host_tensor);
		}
		phi -= (0.0) / width;
		theta -= (movement_sign * 10.0) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
		frame_number++;
	}
	return 0;
}
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        break;
      case GLFW_KEY_SPACE:
        /*camchanged = true;
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        cam.lookAt = ogLookAt;*/
        break;
      }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}

///Example code
//-----------------------------------------------------------------------------------------------------
void load_obj() {
	std::string inputfile = "../scenes/bunny.obj";
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

	if (!warn.empty()) {
		std::cout << warn << std::endl;
	}

	if (!err.empty()) {
		std::cerr << err << std::endl;
	}

	if (!ret) {
		exit(1);
	}

	// Loop over shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
				tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
				tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
				tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
				//tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
				//tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
				// Optional: vertex colors
				// tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
				// tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
				// tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
			}
			index_offset += fv;

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}
}