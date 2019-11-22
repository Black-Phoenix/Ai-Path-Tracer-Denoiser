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
    int loadCamera();
	int loadGeom(string objectid);
	int loadScene(string path, string scene_name);
	int loadMaterial(string materialid);
public:
    Scene(string filename);
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
	std::vector<glm::vec3> textures;
};
