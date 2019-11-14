#pragma once

#include "intersections.h"
#define MESH_NORMAL_VIEW false
#define FRESNELS true

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

__host__ __host__ __device__
bool refract(const glm::vec3& v, const glm::vec3& n, float ni_over_nt, glm::vec3& refracted) {
	glm::vec3 uv = glm::normalize(v);
	float dt = glm::dot(uv, n);
	float discriminat = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminat > 0) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminat);
		return true;
	}
	else
		return false; // no refracted ray
}


__host__ __device__ __inline__ float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx); // ref_idx = n2/n1
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}


__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
		const ShadeableIntersection &intersection,
        const Material &m,
        thrust::default_random_engine &rng) {
	glm::vec3 dir = pathSegment.ray.direction;
	glm::vec3 color(1.0f, 1.0f, 1.0f);
	thrust::uniform_real_distribution<float> u01(0, 1);
	float reflective_prob = m.hasReflective;
	if (reflective_prob != 0 || m.hasRefractive != 0) {
		float pdf = u01(rng), refrac_index_ratio, cosine;
		glm::vec3 normal;
		// Check if it is entry or exit of the object 
		cosine = glm::dot(glm::normalize(dir), intersection.surfaceNormal);
		if (cosine <= 0) { //intersection.is_inside
			normal = intersection.surfaceNormal;
			refrac_index_ratio = 1 / m.indexOfRefraction;
			cosine = -cosine;
		}
		else {
			normal = -intersection.surfaceNormal;
			refrac_index_ratio = m.indexOfRefraction;
		}
		if (FRESNELS) {
			// Check if refraction can occure
			if (refract(pathSegment.ray.direction, normal, refrac_index_ratio, dir))
				// Call schlicks to update the probs
				reflective_prob = schlick(cosine, refrac_index_ratio);
			else
				reflective_prob = 1.0f;
		}
		// Now check if we are going to reflect or refract
		if (pdf < reflective_prob) { 
			dir = glm::normalize(glm::reflect(dir, intersection.surfaceNormal));
			if (MESH_NORMAL_VIEW)
				color = intersection.surfaceNormal;
			else
				color = m.specular.color;
		}
		else {
			dir = glm::normalize(glm::refract(pathSegment.ray.direction, normal, refrac_index_ratio));
			// total internal reflection
			if (!glm::length(dir)) {
				dir = glm::normalize(glm::reflect(dir, intersection.surfaceNormal));
				if (MESH_NORMAL_VIEW)
					color = intersection.surfaceNormal;
				else
					color = m.specular.color;
			}
			else
				if (MESH_NORMAL_VIEW)
					color = intersection.surfaceNormal;
				else
					color = m.color;
		}
	}
	else {
		dir = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng));
		if (MESH_NORMAL_VIEW)
			color = intersection.surfaceNormal;
		else
			color = m.color;
	}
	pathSegment.ray.direction = dir;
	pathSegment.ray.origin = intersection.intersect + dir * 0.01f;
	if (MESH_NORMAL_VIEW)
		pathSegment.color *= glm::abs(color);
	else
		pathSegment.color *= color;//glm::clamp(pathSegment.color * color, glm::vec3(0.0f), glm::vec3(1.0f));
}
