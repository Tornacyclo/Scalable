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

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/IterativeLinearSolvers>

#include <fstream>



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

    Eigen::MatrixXd getEigenMap() const;
    const char* getOutput() const;
    void LocalGlobalParametrization(const char* map);

    Eigen::VectorXd xk;
    Eigen::VectorXd xk_1;
    Eigen::VectorXd pk;
    Eigen::VectorXd dk;

private:
    Triangles mOri;
    Triangles mTut;
    Triangles mLocGlo;
    std::set<int> blade;
    std::set<int> bound;
    std::unordered_map<int, double> fOriMap;
    std::unordered_map<int, double> area;
    std::unordered_map<int, Eigen::Matrix2d> Shape;
    std::unordered_map<int, mat<3,2>> ref_tri;
    Eigen::MatrixXd EigenMap;
    char output_name[120];
    char energy[65] = "arap";
    int max_iterations = 30;

    int num_vertices;
	int num_triangles;
    std::vector<Eigen::Matrix2d> Rot, Jac, Wei, Grad;
    Eigen::MatrixXd Af; // Area factor
    Eigen::SparseMatrix<double> Dx, Dy;
    Eigen::VectorXi dx_i, dx_j;////////////////////////

    double calculateTriangleArea(const vec3& v0, const vec3& v1, const vec3& v2);
    double calculateCotan(const vec3& v0, const vec3& v1, const vec3& v2, const vec3& v3);
    double triangle_area_2d(const vec2& v0, const vec2& v1, const vec2& v2);
    double triangle_aspect_ratio_2d(const vec2& v0, const vec2& v1, const vec2& v2);
    void reference_mesh(Triangles& map);
    void bound_vertices_circle_normalized(Triangles& map);
    void Tut63(const int acount, char** avariable);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> compute_gradients(double u1, double v1, double u2, double v2, double u3, double v3);
    void jacobian_rotation_area(Triangles& map, bool lineSearch);
    void update_weights();
    void least_squares();
    void verify_flips(Triangles& map,
				std::vector<int>& ind_flip);
    int flipsCount(Triangles& map);
    void updateUV(Triangles& map, const Eigen::VectorXd& xk);
    double minimum_step_singularities(Triangles& map, Eigen::VectorXd& x, Eigen::VectorXd& d);
    double smallest_position_quadratic_zero(double a, double b, double c);
    double determineAlphaMax(const Eigen::VectorXd& xk, const Eigen::VectorXd& dk,
											Triangles& map);
    void add_energies_jacobians(double& norm_arap_e, bool flips_linesearch);
    void compute_energy_gradient(Eigen::VectorXd& grad, bool flips_linesearch, Triangles& map);
    void computeGradient(Eigen::VectorXd& x, Eigen::VectorXd& grad, Triangles& map);
    void computeAnalyticalGradient(Eigen::VectorXd& x, Eigen::VectorXd& grad, Triangles& map);
    double lineSearch(Eigen::VectorXd& xk_search, Eigen::VectorXd& dk,
                      Triangles& map);
    void nextStep(Triangles& map);
};


#endif