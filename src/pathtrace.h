#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
// Static env
static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL; 
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_intersections_cache = NULL;
static Face * dev_faces = NULL;
static MeshBoundingBox * dev_mesh_box = NULL;
static float * dev_tensor = NULL; // stores 10 channels here 
