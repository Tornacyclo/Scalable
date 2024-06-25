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
#include <cmath>
#include <functional>



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


// Template for the objective function
template<typename FuncType>
double objectiveFunctionTemplate(const Eigen::VectorXd& x, FuncType objFunc) {
    return objFunc(x);
}

// Template for the gradient function
template<typename GradType>
Eigen::VectorXd gradientFunctionTemplate(const Eigen::VectorXd& x, GradType gradFunc) {
    return gradFunc(x);
}


class TrianglesMapping {
public:
    TrianglesMapping(const int acount, char** avariable);

    Eigen::MatrixXd getEigenMap() const;
    const char* getOutput() const;
    void LocalGlobalParametrization(const char* map);

    Eigen::VectorXd xk;
    Eigen::VectorXd xk_1;
    Eigen::VectorXd pk;

private:
    Triangles mOri;
    Triangles mTut;
    Triangles mLocGlo;
    Eigen::MatrixXd EigenMap;
    char output_name[65];
    char energy[65] = "arap";
    int max_iterations = 100;

    int num_vertices;
	int num_triangles;
    std::vector<Eigen::Matrix2d> Rot, Jac, Wei;
    Eigen::MatrixXd Af; // Area factor
    Eigen::MatrixXd Dx, Dy;
    Eigen::VectorXi dx_i, dx_j;////////////////////////

    double calculateTriangleArea(const vec3& v0, const vec3& v1, const vec3& v2);
    double calculateCotan(const vec3& v0, const vec3& v1, const vec3& v2, const vec3& v3);
    void Tut63(const int acount, char** avariable);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> compute_gradients(double u1, double v1, double u2, double v2, double u3, double v3);
    void jacobian_rotation_area(Triangles& map);
    void update_weights();
    void least_squares();
    double lineSearch(const Eigen::VectorXd& xk, const Eigen::VectorXd& dk, std::function<double(const Eigen::VectorXd&)> objFunc, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradFunc);
    void nextStep();
};


#endif