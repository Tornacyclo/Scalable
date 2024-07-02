#ifndef UV_TO_WORLD_H
#define UV_TO_WORLD_H


#include <eigen3/Eigen/Dense>
#include <ultimaille/all.h>


using namespace UM;



Eigen::Matrix3d uvToWorld(const vec3 &A, const vec3 &B, const vec3 &C, const vec2 &A_prime, const vec2 &B_prime, const vec2 &C_prime);


#endif // UV_TO_WORLD_H