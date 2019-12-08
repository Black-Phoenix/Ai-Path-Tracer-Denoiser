#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();
	int loadObj(string objectid);
    std::vector<Geom> geoms;
    std::vector<Material> materials;
	std::vector<Face> faces;
	MeshBoundingBox mesh_box;
    RenderState state;
	__inline__ void update_mesh_box(glm::vec3 pos) {
		// min
		if (mesh_box.lb.x > pos.x)
			mesh_box.lb.x = pos.x;
		if (mesh_box.lb.y > pos.y)
			mesh_box.lb.y = pos.y;
		if (mesh_box.lb.z > pos.z)
			mesh_box.lb.z = pos.z;
		// max
		if (mesh_box.ub.x < pos.x)
			mesh_box.ub.x = pos.x;
		if (mesh_box.ub.y < pos.y)
			mesh_box.ub.y = pos.y;
		if (mesh_box.ub.z < pos.z)
			mesh_box.ub.z = pos.z;
	}
};
