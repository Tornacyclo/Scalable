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


using namespace UM;


class TrianglesMapping {
public:
    TrianglesMapping(const int acount, char** avariable);

    Triangles getUltiMap() const;
    Eigen::MatrixXf getEigenMap() const;
    Triangles getOriginal() const;
    const char* getOutput() const;
    Triangles LocalGlobalParametrization(Triangles map);

    Eigen::MatrixXd Ri,Ji;

private:
    Triangles mOri;
    Triangles mTut;
    Eigen::MatrixXf EigenMap;
    char output_name[65];

    Eigen::VectorXi dxi,dxj;

    double calculateTriangleArea(const vec3& v0, const vec3& v1, const vec3& v2);
    double calculateCotan(const vec3& v0, const vec3& v1, const vec3& v2, const vec3& v3);
    void Tut63(const int acount, char** avariable);
    void jacobians(const Triangles& map);
};


#endif
