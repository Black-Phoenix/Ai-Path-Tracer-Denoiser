#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }

    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
				loadMaterial(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
				loadGeom(tokens[1]);
				cout << " " << endl;
			} else if (strcmp(tokens[0].c_str(), "SCENE") == 0) {
				loadScene(tokens[1], tokens[2]);
                cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
				loadCamera();
				cout << " " << endl;
			}
		}
    }
	int x;
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
	state.albedos.resize(arraylen);
	state.normals.resize(arraylen);
	state.depth.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
	std::fill(state.albedos.begin(), state.albedos.end(), glm::vec3());
	std::fill(state.normals.begin(), state.normals.end(), glm::vec3());
	std::fill(state.depth.begin(), state.depth.end(), 0.0f);
    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadGeom(string objectid) {
	int id = atoi(objectid.c_str());
	if (id != geoms.size()) {
		cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
		return -1;
	}
	else {
		cout << "Loading Geom " << id << "..." << endl;
		Geom newGeom;
		string line;

		//load object type
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			if (strcmp(line.c_str(), "sphere") == 0) {
				cout << "Creating new sphere..." << endl;
				newGeom.type = SPHERE;
			}
			else if (strcmp(line.c_str(), "cube") == 0) {
				cout << "Creating new cube..." << endl;
				newGeom.type = CUBE;
			}
		}

		//link material
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			newGeom.materialid = atoi(tokens[1].c_str());
			cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
		}

		//load transformations & vel
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);

			//load tranformations
			if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "VEL") == 0) {
				newGeom.vel = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			utilityCore::safeGetline(fp_in, line);
		}

		newGeom.transform = utilityCore::buildTransformationMatrix(
			newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

		geoms.push_back(newGeom);
		return 1;
	}
}

int Scene::loadMaterial(string materialid) {
	int id = atoi(materialid.c_str());
	if (id != materials.size()) {
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}
	else {
		cout << "Loading Material " << id << "..." << endl;
		Material newMaterial;

		//load static properties
		for (int i = 0; i < 7; i++) {
			string line;
			utilityCore::safeGetline(fp_in, line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "RGB") == 0) {
				glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.color = color;
			}
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
				newMaterial.specular.exponent = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
				glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.specular.color = specColor;
			}
			else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
				newMaterial.reflective = false;//atof(tokens[1].c_str()) > 0.5;
				newMaterial.diffused = true;
			}
			else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
				newMaterial.refractive = false;// atof(tokens[1].c_str()) > 0.5;
			}
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
				newMaterial.emittance = glm::vec3(atof(tokens[1].c_str()));
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}

glm::vec3 calculate_geometric_normals(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2) {
	glm::vec3 edge10 = p0 - p1; //p1 - p0;

	glm::vec3 edge20 = p0 - p2; // p2 - p0;

	return glm::normalize(glm::cross(edge10, edge20));
}

int Scene::loadScene(string path, string scene_name) {
	//Load transform
	string line;
	utilityCore::safeGetline(fp_in, line);
	glm::vec3 translation, rotation, scale, vel;
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);

		//load tranformations
		if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
			translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
			rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
			scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "VEL") == 0) {
			vel = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		utilityCore::safeGetline(fp_in, line);
	}

	glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
	glm::mat4 inverseTransform = glm::inverse(transform);
	glm::mat4 invTranspose = glm::inverseTranspose(transform);
	// Load obj
	tinyobj::attrib_t obj_attrib;
	std::vector<tinyobj::shape_t> obj_shapes;
	std::vector<tinyobj::material_t> obj_materials;

	std::string warn;
	std::string err;

	bool ret = tinyobj::LoadObj(&obj_attrib, &obj_shapes, &obj_materials, &warn, &err, (path + scene_name).c_str(), path.c_str());

	if (!warn.empty()) {
		std::cout << warn << std::endl;
	}

	if (!err.empty()) {
		std::cerr << err << std::endl;
		return 1;
	}

	if (!ret) {
		cout << "Larger failure" << endl;
		return 1;
	}
	// Reserver materials
	int material_offset = materials.size();  // use this to deal with other objects in the scene
	materials.reserve(obj_materials.size());
	// Process Materials
	for (int m = 0; m < obj_materials.size(); m++) {
		// Init new material
		Material newMaterial;
		//newMaterial.diffused = false;
		//newMaterial.reflective = false;
		//newMaterial.fresnels = false;
		//newMaterial.glass = false;
		//newMaterial.refractive = false;

		auto material = obj_materials[m];
		// Materials
		if (material.diffuse_texname != "") { // has something
			string text_name = path + material.diffuse_texname;
			cv::Mat image = cv::imread(text_name.c_str());
			if (!image.data) 
				cout << "Couldn't open texture at " << text_name << endl;
			else {
				newMaterial.col = image.cols;
				newMaterial.row = image.rows;
				newMaterial.texture_offset = textures.size();
				for (int i = 0; i < newMaterial.row; i++) {
					for (int j = 0; j < newMaterial.col; j++) {
						cv::Vec3b &rbg = image.at<cv::Vec3b>(i, j);
						// Read data into large array
						glm::vec3 texture;
						texture[0] = rbg[0] / 255.0;
						texture[1] = rbg[1] / 255.0;
						texture[2] = rbg[2] / 255.0;
						textures.push_back(texture);
					}
				}
			}
		}
		// color
		newMaterial.color = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
		// specular
		newMaterial.specular.color = glm::vec3(material.specular[0], material.specular[1], material.specular[2]);
		// Illuminitation type (All have specular...)
		if (material.illum < 3 || material.illum == 10) // then diffused
			newMaterial.diffused = true;
		else if (material.illum == 3 || material.illum == 6) {
			newMaterial.reflective = true;
			newMaterial.fresnels = false;
		}
		else if (material.illum == 4 || material.illum == 9)
			newMaterial.glass = true;
		else if (material.illum == 5) {
			newMaterial.reflective = true;
			newMaterial.fresnels = true;
		}
		else if (material.illum == 7) {
			newMaterial.fresnels = true;
			newMaterial.refractive = true;
		}
		// indexOfRefraction
		newMaterial.indexOfRefraction = material.ior;
		// emittance
		newMaterial.emittance = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);
		// roughness
		newMaterial.roughness = material.roughness;
		// specular exp
		newMaterial.specular.exponent = material.shininess;
		// save
		materials.push_back(newMaterial);
	}
	// Process Shapes
	for (size_t s = 0; s < obj_shapes.size(); s++) {
		auto shape = obj_shapes[s];
		size_t index_offset = 0;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
			int fv = shape.mesh.num_face_vertices[f];
			assert(fv == 3);
			Geom newGeom;
			newGeom.type = TRIANGLE;
			newGeom.materialid = material_offset + shape.mesh.material_ids[f];
			newGeom.translation = translation;
			newGeom.rotation = rotation;
			newGeom.scale = scale;
			newGeom.transform = transform;
			newGeom.inverseTransform = inverseTransform;
			newGeom.invTranspose = invTranspose;
			newGeom.vel = vel;
			// TODO Set translation, transformation, etc
			// Load attributes (n, v,and uv)
			for (size_t v = 0; v < fv; v++) {
				tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
				newGeom.t.vs[v] = glm::vec3(obj_attrib.vertices[3 * idx.vertex_index + 0], 
											obj_attrib.vertices[3 * idx.vertex_index + 1], 
											obj_attrib.vertices[3 * idx.vertex_index + 2]);
				newGeom.t.vs[v] = glm::vec3(transform * glm::vec4(newGeom.t.vs[v], 1));
				if (obj_attrib.normals.size() != 0) {
					// use existing normals
					newGeom.t.ns[v] = glm::vec3(obj_attrib.normals[3 * idx.normal_index + 0],
												obj_attrib.normals[3 * idx.normal_index + 1],
												obj_attrib.normals[3 * idx.normal_index + 2]);
					newGeom.t.ns[v] = glm::vec3(invTranspose * glm::vec4(newGeom.t.ns[v], 0));
				}
				if (obj_attrib.texcoords.size() != 0) {
					// load texture coordinates
					newGeom.t.uvs[v] = glm::vec2(obj_attrib.texcoords[2 * idx.texcoord_index + 0],
												 obj_attrib.texcoords[2 * idx.texcoord_index + 1]);
				}
			}
			if (obj_attrib.normals.size() == 0) {
				// recompute normals
				newGeom.t.ns[0] = newGeom.t.ns[1] = newGeom.t.ns[2] = calculate_geometric_normals(newGeom.t.vs[0], newGeom.t.vs[1], newGeom.t.vs[2]);
			}
			index_offset += fv;
			geoms.push_back(newGeom);
		}
	}
	cout << "Loaded Mesh" << endl;
}
