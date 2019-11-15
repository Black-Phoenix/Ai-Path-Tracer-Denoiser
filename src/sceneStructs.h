#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
	TRIANGLE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
	// Vertices
	glm::vec3 vs[3];
	// Normals
	glm::vec3 ns[3];
	// texture coordinates
	glm::vec3 uvs[3];
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
	glm::vec3 vel;
	Triangle t;
};



struct Face {
	glm::vec3 v[3];
	glm::vec3 n[3];
	int materialid;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float indexOfRefraction;
    glm::vec3 emittance;
	float roughness;
	// bools
	bool diffused;
	bool fresnels;
	bool glass;
	bool reflective;
	bool refractive;
	Material(){
		diffused = false;
		fresnels = false;
		glass = false;
		reflective = false;
		refractive = false;
	}
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
	std::vector<glm::vec3> normals;
	std::vector<float> depth;
	std::vector<glm::vec3> albedos;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

struct MeshBoundingBox {
	glm::vec3 lb;
	glm::vec3 ub;
};
// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  bool is_inside;
  glm::vec3 intersect;
};
