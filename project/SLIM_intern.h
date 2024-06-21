#ifndef SLIM_INTERN_H
#define SLIM_INTERN_H


#include "helpers.h"
#include <ultimaille/all.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <map>
#include <set>
#include <filesystem>
#include <stdlib.h>
#include <string>

#include <iostream>
#include <thread>
#include <chrono>



#ifdef linux
	#define DBOUT( s )\
	{\
	std::cout << s;\
	}
#endif

#ifdef _WIN32
	#define DBOUT( s )\
	{\
	   std::ostringstream os_;\
	   os_ << s;\
	   OutputDebugString( os_.str().c_str() );\
	}
#endif


using namespace UM;


class TrianglesMapping {
public:
    TrianglesMapping(const int acount, char** avariable);

    Eigen::MatrixXf getEigenMap() const;
    const char* getOutput() const;
    void LocalGlobalParametrization(Triangles& map);

    std::vector<Eigen::Matrix2d> Rot, Jac, Wei;
    Eigen::MatrixXd Dx, Dy;

private:
    Triangles mOri;
    Triangles mTut;
    Triangles mLocGlo;
    Eigen::MatrixXf EigenMap;
    char output_name[65];
    char energy[65];

    Eigen::VectorXi dx_i, dx_j;////////////////////////

    double calculateTriangleArea(const vec3& v0, const vec3& v1, const vec3& v2);
    double calculateCotan(const vec3& v0, const vec3& v1, const vec3& v2, const vec3& v3);
    void Tut63(const int acount, char** avariable);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> TrianglesMapping::compute_gradients(double u1, double v1, double u2, double v2, double u3, double v3);
    void jacobians(Triangles& map);
    void update_weights();
};


#endif
