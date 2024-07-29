#ifndef SLIM_INTERN_H
#define SLIM_INTERN_H


#include "helpers.h"
#include <ultimaille/all.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <filesystem>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Sparse>
#include <fstream>

#include "igl/grad.h"
#include "igl/local_basis.h"
#include "igl/read_triangle_mesh.h"
#include "igl/polar_svd.h"
#include "igl/flip_avoiding_line_search.h"
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>



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

    Eigen::MatrixXd uv;
    Eigen::MatrixXd uv_1;
    Eigen::MatrixXd distance;

private:
    Triangles mOri;
    Triangles mTut;
    Triangles mLocGlo;
    std::set<int> blade;
    std::set<int> bound;
    std::vector<int> bound_sorted;
    std::unordered_map<int, double> fOriMap;
    std::unordered_map<int, double> area;
    std::unordered_map<int, Eigen::Matrix2d> Shape_1;
    std::unordered_map<int, mat<3,2>> ref_tri;
    bool first_time = true;
    std::vector<double> distortion_energy;
    Eigen::MatrixXd EigenMap;
    char output_name_geo[250];
    char output_name_obj[250];
    char times_txt[150];
    int num_vertices;
	int num_triangles;
    Eigen::MatrixXd V; // card(V) by 3, list of mesh vertex positions
    Eigen::MatrixXi F; // card(F) by 3/3, list of mesh faces (triangles/tetrahedra)
    Eigen::MatrixXd V_1;
    Eigen::MatrixXi F_1;
    Eigen::MatrixXd Ri, Ji;
    std::vector<Eigen::Matrix2d> Rot, Jac, Wei;
    Eigen::MatrixXd Af;
    Eigen::SparseMatrix<double> D_x, D_y, D_z;
    double lambda = 1e-4;
    Eigen::VectorXd flattened_weight_matrix;
    Eigen::VectorXd mass;
    double weight_option = 1.0;
    double exponential_factor = 1e-3;
    Eigen::VectorXd rhs;
    double alpha;
    Eigen::VectorXd M;
    double mesh_area;
    double energumene;
    double lambda_polyconvex = 1;
    double epsilon;
    int dimension = 2;
    const char* energy;
    int max_iterations;
    
    std::chrono::high_resolution_clock::time_point totalStart;
    long long totalTime;

    double calculateTriangleArea(const vec3& v0, const vec3& v1, const vec3& v2);
    double calculateCotan(const vec3& v0, const vec3& v1, const vec3& v2, const vec3& v3);
    double triangle_area_2d(const vec2& v0, const vec2& v1, const vec2& v2);
    double triangle_aspect_ratio_2d(const vec2& v0, const vec2& v1, const vec2& v2);
    void reference_mesh(Triangles& map);
    void bound_vertices_circle_normalized(Triangles& map);
    void map_vertices_to_circle_area_normalized(Triangles& map, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXi& bnd, Eigen::MatrixXd& UV);
    void Tut63(const char* name, int weights);
    void jacobian_rotation_area(Triangles& map, bool lineSearch);
    void least_squares();
    void updateUV(Triangles& map, const Eigen::VectorXd& xk);
    void fillUV(Eigen::MatrixXd& V_new, const Eigen::VectorXd& xk);
    double step_singularities(const Eigen::MatrixXi& F, const Eigen::MatrixXd& uv, const Eigen::MatrixXd& d);
    double smallest_position_quadratic_zero(double a, double b, double c);
    double determineAlphaMax(const Eigen::VectorXd& xk, const Eigen::VectorXd& dk,
											Triangles& map);
    double add_energies_jacobians(Eigen::MatrixXd& V_new, bool flips_linesearch);
    void compute_energy_gradient(Eigen::VectorXd& grad, bool flips_linesearch, Triangles& map);
    double lineSearch(Eigen::MatrixXd& xk_current, Eigen::MatrixXd& dk, Triangles& map);
    void nextStep(Triangles& map);
};


#endif // #ifndef SLIM_INTERN_H