/**
 * Scalable Locally Injective Mappings
*/

#include "SLIM_intern.h"



// When using Graphite:
// - For UltiMaille, v.pos()[0] is the x-coordinate, v.pos()[1] is the z-coordinate and v.pos()[2] is the y-coordinate
// - For libigl, V.col(0) is the x-coordinate, V.col(1) is the z-coordinate and V.col(2) is the y-coordinate
// For example, in an .obj file, v -1.000000 0.866000 0.000000, the x-coordinate is -1.000000, the z-coordinate is 0.866000 and the y-coordinate is 0.000000
TrianglesMapping::TrianglesMapping(const int acount, char** avariable) {
    const char* name = nullptr;
    int weights = -1;
    max_iterations = -1;
    orientation = nullptr;
    energy = nullptr;

    if (acount > 1) {
        for (int i = 1; i < acount; ++i) {
            if (strlen(avariable[i]) == 1 && isdigit(avariable[i][0])) {
                weights = atoi(avariable[i]);
            } else if (strlen(avariable[i]) > 1) {
                if (strcmp(avariable[i], "xyz") == 0 || strcmp(avariable[i], "xzy") == 0 || strcmp(avariable[i], "yxz") == 0 || strcmp(avariable[i], "yzx") == 0 || strcmp(avariable[i], "zxy") == 0 || strcmp(avariable[i], "zyx") == 0) {
                    orientation = avariable[i];
                }
                else if (strcmp(avariable[i], "ARAP") == 0 || strcmp(avariable[i], "SYMMETRIC-DIRICHLET") == 0 || strcmp(avariable[i], "EXPONENTIAL-SYMMETRIC-DIRICHLET") == 0 || strcmp(avariable[i], "HENCKY-STRAIN") == 0 || strcmp(avariable[i], "AMIPS") == 0 || strcmp(avariable[i], "CONFORMAL-AMIPS-2D") == 0 || strcmp(avariable[i], "CONFORMAL-AMIPS-3D") == 0 || strcmp(avariable[i], "UNTANGLE-2D") == 0) {
                    energy = avariable[i];
                }
                else if (strncmp(avariable[i], "max_iterations=", 15) == 0) {
                    max_iterations = atoi(avariable[i] + 15);
                }
                else {
                    name = avariable[i];
                }
            }
        }
    }

    if (name == nullptr) {
        #ifdef _WIN32
            name = "mesh_test/hemisphere.obj";
        #endif
        #ifdef linux
            name = "project/mesh_test/hemisphere.obj";
        #endif
    }

    if (orientation == nullptr) {
        orientation = "xyz";
    }

    if (energy == nullptr) {
        energy = "ARAP";
    }

    if (weights == -1) {
        weights = 2;
    }
    
    if (max_iterations == -1) {
        max_iterations = 100;
    }

    std::cout << "Name: " << name << std::endl;
    std::cout << "Orientation: " << orientation << std::endl;
    std::cout << "Energy: " << energy << std::endl;
    std::cout << "Weights: " << weights << std::endl;
    std::cout << "Max Iterations: " << max_iterations << std::endl;

    Tut63(name, weights);
}

Eigen::MatrixXd TrianglesMapping::getEigenMap() const {
	return EigenMap;
}

const char* TrianglesMapping::getOutput() const {
    return output_name_geo;
}

double TrianglesMapping::calculateTriangleArea(const vec3& v0, const vec3& v1, const vec3& v2) {
    vec3 side1 = v1 - v0;
    vec3 side2 = v2 - v0;
    vec3 crossProduct = cross(side1, side2);
    double area = crossProduct.norm() / 2.0;
    return area;
}

double TrianglesMapping::calculateCotan(const vec3& v0, const vec3& v1, const vec3& v2, const vec3& v3) {
    vec3 v = v0 - v1;
    vec3 w = v2 - v3;
    double cotan = v * w / cross(v, w).norm();
    return cotan;
}

double TrianglesMapping::triangle_area_2d(const vec2& v0, const vec2& v1, const vec2& v2) {
    return 0.5 * ((v1.y - v0.y) * (v1.x + v0.x) + (v2.y - v1.y) * (v2.x + v1.x) + (v0.y - v2.y) * (v0.x + v2.x));
}

double TrianglesMapping::triangle_aspect_ratio_2d(const vec2& v0, const vec2& v1, const vec2& v2) {
    double l1 = (v1 - v0).norm();
    double l2 = (v2 - v1).norm();
    double l3 = (v0 - v2).norm();
    double lmax = std::max(l1, std::max(l2, l3));
    return lmax * (l1 + l2 + l3) / (4.0 * std::sqrt(3.0) * triangle_area_2d(v0, v1, v2));
}

void TrianglesMapping::reference_mesh(Triangles& map) {
    for (int f : facet_iter(map)) {
            // area[int(f)] = map.util.unsigned_area(f); A DEBOGUER
            vec2 A,B,C;
            // map.util.project(f, A, B, C); A DEBOGUER

            double ar = triangle_aspect_ratio_2d(A, B, C);
            if (ar>10) { // If the aspect ratio is bad, assign an equilateral reference triangle
                double a = ((B-A).norm() + (C-B).norm() + (A-C).norm())/3.; // Edge length is the average of the original triangle
                area[int(f)] = sqrt(3.)/4.*a*a;
                A = {0., 0.};
                B = {a, 0.};
                C = {a/2., std::sqrt(3.)/2.*a};
            }

            mat<2,2> ST = {{B-A, C-A}};
            ref_tri[int(f)] = mat<3,2>{{ {-1,-1},{1,0},{0,1} }}*ST.invert_transpose();

            Eigen::Matrix2d S;
            S << B.x - A.x, C.x - A.x,
                 B.y - A.y, C.y - A.y;
            Shape_1[int(f)] = S.inverse();
    }
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> TrianglesMapping::compute_gradients(double u1, double v1, double u2, double v2, double u3, double v3) {
    double A = 0.5 * std::fabs(u1 * v2 + u2 * v3 + u3 * v1 - u1 * v3 - u2 * v1 - u3 * v2);
    Eigen::Vector3d dudN, dvdN;
    dudN << (v2 - v3) / (2 * A), (v3 - v1) / (2 * A), (v1 - v2) / (2 * A);
    dvdN << (u3 - u2) / (2 * A), (u1 - u3) / (2 * A), (u2 - u1) / (2 * A);
    return std::make_pair(dudN, dvdN);
}

void compute_surface_gradient_matrix(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &F1,
                                    const Eigen::MatrixXd &F2, Eigen::SparseMatrix<double> &D1, Eigen::SparseMatrix<double> &D2) {
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);
    Eigen::SparseMatrix<double> Dx = G.block(0, 0, F.rows(), V.rows());
    Eigen::SparseMatrix<double> Dy = G.block(F.rows(), 0, F.rows(), V.rows());
    Eigen::SparseMatrix<double> Dz = G.block(2 * F.rows(), 0, F.rows(), V.rows());

    D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
    D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;
}

void TrianglesMapping::jacobian_rotation_area(Triangles& map, bool lineSearch) {
    num_vertices = map.nverts();
    num_triangles = map.nfacets();
    if (!lineSearch) {xk_1 = Eigen::VectorXd::Zero(2 * num_vertices);}
    // if (first_time) {Af = Eigen::MatrixXd::Zero(num_triangles, num_triangles);}

    Jac.clear();
    Rot.clear();
    int ind = 0;

    Eigen::SparseMatrix<double> Dx_faux, Dy_faux;
    Dx_faux = Eigen::SparseMatrix<double>(num_triangles, num_vertices);
    Dy_faux = Eigen::SparseMatrix<double>(num_triangles, num_vertices);
    std::vector<Eigen::Triplet<double>> Dx_triplets;
    std::vector<Eigen::Triplet<double>> Dy_triplets;

    for (auto f : map.iter_facets()) {
        Eigen::Matrix2d J_i;
        Eigen::Matrix<double, 3, 2> Z_i;
        Z_i << -1, -1,
               1, 0,
               0, 1;
        
        Z_i *= Shape_1[int(f)];

        // std::cout << "Z_i: " << std::endl << Z_i << std::endl;
        // std::cout << "Z_i_ref: " << std::endl << ref_tri[int(f)] << std::endl;

        Dx_triplets.push_back(Eigen::Triplet<double>(ind, int(f.vertex(0)), Z_i(0, 0) + Z_i(0, 1)));
        Dx_triplets.push_back(Eigen::Triplet<double>(ind, int(f.vertex(1)), Z_i(1, 0) + Z_i(1, 1)));
        Dx_triplets.push_back(Eigen::Triplet<double>(ind, int(f.vertex(2)), Z_i(2, 0) + Z_i(2, 1)));
        Dy_triplets.push_back(Eigen::Triplet<double>(ind, int(f.vertex(0)), Z_i(0, 0) + Z_i(0, 1)));
        Dy_triplets.push_back(Eigen::Triplet<double>(ind, int(f.vertex(1)), Z_i(1, 0) + Z_i(1, 1)));
        Dy_triplets.push_back(Eigen::Triplet<double>(ind, int(f.vertex(2)), Z_i(2, 0) + Z_i(2, 1)));

        for (int j = 0; j < 3; ++j) {
            int v_ind = int(f.vertex(j));
            if (!lineSearch) {
                xk_1(v_ind) = f.vertex(j).pos()[0];
                xk_1(v_ind + num_vertices) = f.vertex(j).pos()[1];
            }
        }

        J_i(0, 0) = f.vertex(1).pos()[0] - f.vertex(0).pos()[0];
        J_i(1, 0) = f.vertex(1).pos()[2] - f.vertex(0).pos()[2];
        J_i(0, 1) = f.vertex(2).pos()[0] - f.vertex(0).pos()[0];
        J_i(1, 1) = f.vertex(2).pos()[2] - f.vertex(0).pos()[2];

        J_i *= Shape_1[int(f)];

        // std::cout << "J_i: " << std::endl << J_i << std::endl;
        Jac.push_back(J_i);
        // Compute SVD of J_i
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(J_i, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d U = svd.matrixU();
        Eigen::Matrix2d V = svd.matrixV();
        /*Eigen::Matrix2d R_i;
        if ((U * V.transpose()).determinant() > 0) {
        // Construct the closest rotation matrix R_i
            R_i = U * V.transpose();
        } else {
            Eigen::Vector2d singu = svd.singularValues();
                double s1 = singu(0);
                double s2 = singu(1);
                double singu_min = std::min(s1, s2);
                Eigen::Matrix2d singu_Mat = Eigen::Matrix2d::Identity();
                if (singu_min == s1) {
                    singu_Mat(0, 0) = -1;
                }
                else if (singu_min == s2) {
                    singu_Mat(1, 1) = -1;
                }
                else {
                    singu_Mat(0, 0) = -1;
                }
            // V.col(1) *= -1;
            R_i = U*singu_Mat*V.transpose();
        }*/

        Eigen::Matrix2d R_i;
        R_i = U * V.transpose();

        /*if ((U * V.transpose()).determinant() > 0) {
            // Construct the closest rotation matrix R_i
            R_i = U * V.transpose();
        } else {
            // Adjust the sign of the last column of U or V
            U.col(1) *= -1;
            R_i = U * V.transpose();
        }*/

        Rot.push_back(R_i);

        if (first_time) {
                // Af(ind, ind) = std::sqrt(calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()));
                // Af(ind, ind) = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
        }
        // std::cout << Af(ind, ind) << std::endl;

        ind++;
    }

    // Assemble the sparse matrices Dx and Dy
    Dx_faux.setFromTriplets(Dx_triplets.begin(), Dx_triplets.end());
    Dy_faux.setFromTriplets(Dy_triplets.begin(), Dy_triplets.end());
    // std::cout << "C'EST FAUX\n" << Dx_faux << std::endl;

    Jac.clear();
    Rot.clear();
    if (!lineSearch) {Wei.clear();}
    
    // std::cout << "C'EST JUSTE\n" << Dx << std::endl;
    // Ji=[D1*u, D2*u, D1*v, D2*v];
    Ji = Eigen::MatrixXd::Zero(num_triangles, dimension * dimension);
    Ji.col(0) = Dx * V_1.col(0);
    Ji.col(1) = Dy * V_1.col(0);
    Ji.col(2) = Dx * V_1.col(1);
    Ji.col(3) = Dy * V_1.col(1);

    for (int i = 0; i < Ji.rows(); ++i) {
        Eigen::Matrix<double, 2, 2> ji, ri, ti, ui, vi;
        Eigen::Matrix<double, 2, 1> sing;
        Eigen::Matrix<double, 2, 1> closest_sing_vec;
        Eigen::Matrix<double, 2, 2> mat_W;
        Eigen::Matrix<double, 2, 1> m_sing_new;
        double s1, s2;

        ji(0, 0) = Ji(i, 0);
        ji(0, 1) = Ji(i, 1);
        ji(1, 0) = Ji(i, 2);
        ji(1, 1) = Ji(i, 3);

        igl::polar_svd(ji, ri, ti, ui, sing, vi);

        s1 = sing(0);
        s2 = sing(1);

        Jac.push_back(ji);
        Rot.push_back(ri);

        if (!lineSearch) {
            if (strcmp(energy, "ARAP") == 0) {
                m_sing_new << 1, 1;
            }
            else if (strcmp(energy, "SYMMETRIC-DIRICHLET") == 0) {
                double s1_g = 2 * (s1 - pow(s1, -3));
                double s2_g = 2 * (s2 - pow(s2, -3));
                m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
            }
            else if (strcmp(energy, "EXPONENTIAL-SYMMETRIC-DIRICHLET") == 0) {
                double s1_g = 2 * (s1 - pow(s1, -3));
                double s2_g = 2 * (s2 - pow(s2, -3));

                double in_exp = exponential_factor * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
                double exponential_term = exp(in_exp);

                s1_g *= exponential_term * exponential_factor;
                s2_g *= exponential_term * exponential_factor;

                m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
            }
            else if (strcmp(energy, "HENCKY-STRAIN") == 0) {
                double s1_g = 2 * (log(s1) / s1);
                double s2_g = 2 * (log(s2) / s2);
                m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
            }
            else if (strcmp(energy, "AMIPS") == 0) {
                //
            }
            else if (strcmp(energy, "CONFORMAL-AMIPS-2D") == 0) {
                //
            }
            else if (strcmp(energy, "CONFORMAL-AMIPS-3D") == 0) {
                //
            }
            else if (strcmp(energy, "UNTANGLE-2D") == 0) {
                //
            }

            if (std::abs(s1 - 1) < 1e-8) m_sing_new(0) = 1;
            if (std::abs(s2 - 1) < 1e-8) m_sing_new(1) = 1;
            mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();
            Wei.push_back(mat_W);
        }
    }

    first_time = false;
}

void TrianglesMapping::update_weights() {
	Wei.clear();
	if (strcmp(energy, "ARAP") == 0) {
		for (size_t i = 0; i < Rot.size(); ++i) {
			Wei.push_back(Eigen::Matrix2d::Identity());
		}
	}
	else if (strcmp(energy, "SYMMETRIC-DIRICHLET") == 0) {
		//
	}
    else if (strcmp(energy, "EXPONENTIAL-SYMMETRIC-DIRICHLET") == 0) {
		//
	}
    else if (strcmp(energy, "HENCKY-STRAIN") == 0) {
		//
	}
    else if (strcmp(energy, "AMIPS") == 0) {
		//
	}
    else if (strcmp(energy, "CONFORMAL-AMIPS-2D") == 0) {
		//
	}
    else if (strcmp(energy, "CONFORMAL-AMIPS-3D") == 0) {
		//
	}
	else if (strcmp(energy, "UNTANGLE-2D") == 0) {
		//
	}
}

void TrianglesMapping::least_squares() {
	// Eigen::SparseMatrix<double> A(4 * num_triangles, 2 * num_vertices);
    Eigen::SparseMatrix<double> A(dimension * dimension * num_triangles, dimension * num_vertices);
	//  Eigen::VectorXd b = Eigen::VectorXd::Zero(4 * num_triangles);
    Eigen::VectorXd b(dimension * dimension * num_triangles);
    b.setZero();
    
	// Create diagonal matrices W11, W12, W21, W22 as vectors
	Eigen::VectorXd W11_diag = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd W12_diag = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd W21_diag = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd W22_diag = Eigen::VectorXd::Zero(num_triangles);

	// R vectors
	Eigen::VectorXd R11 = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd R12 = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd R21 = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd R22 = Eigen::VectorXd::Zero(num_triangles);

    std::cout << "Yo long time no see" << std::endl;

	for (int i = 0; i < num_triangles; ++i) {
		W11_diag(i) = Wei[i](0, 0);
		W12_diag(i) = Wei[i](0, 1);
		W21_diag(i) = Wei[i](1, 0);
		W22_diag(i) = Wei[i](1, 1);

		R11(i) = Rot[i](0, 0);
		R12(i) = Rot[i](0, 1);
		R21(i) = Rot[i](1, 0);
		R22(i) = Rot[i](1, 1);
	}

    std::vector<Eigen::Triplet<double>> triplet;
    triplet.reserve(4 * (Dx.outerSize() + Dy.outerSize()));
    // Fill in the A matrix
    /*A = [W11*Dx, W12*Dx;
           W11*Dy, W12*Dy;
           W21*Dx, W22*Dx;
           W21*Dy, W22*Dy];*/
    for (int k = 0; k < Dx.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Dx, k); it; ++it) {
        int dx_row = it.row();
        int dx_col = it.col();
        double val = it.value();

        triplet.push_back(Eigen::Triplet<double>(dx_row, dx_col, val * W11_diag(dx_row)));
        triplet.push_back(Eigen::Triplet<double>(dx_row, num_vertices + dx_col, val * W12_diag(dx_row)));

        triplet.push_back(Eigen::Triplet<double>(2 * num_triangles + dx_row, dx_col, val * W21_diag(dx_row)));
        triplet.push_back(Eigen::Triplet<double>(2 * num_triangles + dx_row, num_vertices + dx_col, val * W22_diag(dx_row)));
        }
    }
    
    for (int k = 0; k < Dy.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Dy, k); it; ++it) {
        int dy_row = it.row();
        int dy_col = it.col();
        double val = it.value();

        triplet.push_back(Eigen::Triplet<double>(num_triangles + dy_row, dy_col, val * W11_diag(dy_row)));
        triplet.push_back(Eigen::Triplet<double>(num_triangles + dy_row, num_vertices + dy_col, val * W12_diag(dy_row)));

        triplet.push_back(Eigen::Triplet<double>(3 * num_triangles + dy_row, dy_col, val * W21_diag(dy_row)));
        triplet.push_back(Eigen::Triplet<double>(3 * num_triangles + dy_row, num_vertices + dy_col, val * W22_diag(dy_row)));
        }
    }

	A.setFromTriplets(triplet.begin(), triplet.end());
    Eigen::SparseMatrix<double> At = A.transpose();
    At.makeCompressed();

    Eigen::SparseMatrix<double> identity_dimA(At.rows(), At.rows());
    identity_dimA.setIdentity();

    // Add a proximal term
    flattened_weight_matrix.resize(dimension * dimension * num_triangles);
    mass.resize(num_triangles);
	mass.setConstant(weight_option); // All the weights are equal to 1
    for (int i = 0; i < dimension * dimension; i++)
        for (int j = 0; j < num_triangles; j++)
            flattened_weight_matrix(i * num_triangles + j) = mass(j);
    Eigen::SparseMatrix<double> L;
    L = At * flattened_weight_matrix.asDiagonal() * A + lambda * identity_dimA;
    L.makeCompressed();

	// Fill in the b vector
    /*b = [W11*R11 + W12*R21;
           W11*R12 + W12*R22;
           W21*R11 + W22*R21;
           W21*R12 + W22*R22];*/
    for (int i = 0; i < num_triangles; i++) {
        b(i + 0 * num_triangles) = W11_diag(i) * R11(i) + W12_diag(i) * R21(i);
        b(i + 1 * num_triangles) = W11_diag(i) * R12(i) + W12_diag(i) * R22(i);
        b(i + 2 * num_triangles) = W21_diag(i) * R11(i) + W22_diag(i) * R21(i);
        b(i + 3 * num_triangles) = W21_diag(i) * R12(i) + W22_diag(i) * R22(i);
    }

    Eigen::VectorXd uv_flat_1(dimension * num_vertices);
        for (int j = 0; j < num_vertices; j++) {
            uv_flat_1(0 * num_vertices + j) = V_1(j, 0);
            uv_flat_1(1 * num_vertices + j) = V_1(j, 1);
        }
    
    rhs.resize(dimension * num_vertices);
    // rhs = (At * flattened_weight_matrix.asDiagonal() * b + lambda * uv_flat_1);
    rhs = (At * flattened_weight_matrix.asDiagonal() * b + lambda * xk_1);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    xk = solver.compute(L).solve(rhs);
    if(solver.info() != Eigen::Success) {
		// Solving failed
		std::cerr << "Solving failed" << std::endl;
		return;
	}

	// Use an iterative solver
	/*Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> solver; // ConjugateGradient solver for symmetric positive definite matrices
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver; // BiCGSTAB solver for square matrices	
    std::cout << W11_diag << std::endl;
    Eigen::SparseMatrix<double> AtA = A.transpose() * A + lambda * Eigen::MatrixXd::Identity(2 * num_vertices, 2 * num_vertices);
    solver.compute(AtA);
    std::cout << "2" << std::endl;
    // solver.compute(A.transpose() * A);

	if(solver.info() != Eigen::Success) {
		// Decomposition failed
		std::cerr << "Decomposition failed" << std::endl;
		return;
	}
    std::cout << "3.5" << std::endl;
    // xk = solver.solve(A.transpose() * b);

    std::cout << "A.transpose() dimensions: " << A.transpose().rows() << " x " << A.transpose().cols() << std::endl;
    std::cout << "b dimensions: " << b.rows() << " x " << b.cols() << std::endl;
    std::cout << "xk_1 dimensions: " << xk_1.rows() << " x " << xk_1.cols() << std::endl;

    std::cout << "num_vertices " << num_vertices * 2 << std::endl;

    Eigen::MatrixXd Atb = A.transpose() * b + lambda * xk_1;
    std::cout << "3" << std::endl;
	xk = solver.solve(Atb);
    std::cout << "4" << std::endl;

	if(solver.info() != Eigen::Success) {
		// Solving failed
		std::cerr << "Solving failed" << std::endl;
		return;
	}*/

    /*Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver; // SimplicialLLT solver for square matrices
    std::cout << "1" << std::endl;
    double lambda = 0.0001;
    Eigen::SparseMatrix<double> AtA = A.transpose() * A + lambda * Eigen::MatrixXd::Identity(A.cols(), A.cols()).sparseView();
    solver.compute(AtA);
    std::cout << "2" << std::endl;

    if(solver.info() != Eigen::Success) {
        // Decomposition failed
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }

    Eigen::MatrixXd Atb = A.transpose() * b + lambda * xk_1;
    std::cout << Atb << std::endl;
    xk = solver.solve(Atb);
    std::cout << "4" << std::endl;

    if(solver.info() != Eigen::Success) {
        // Solving failed
        std::cerr << "Solving failed" << std::endl;
        return;
    }*/

	dk = xk - uv_flat_1;
	// std::cout << "Distance dk:" << std::endl << V_1(5, 1) << std::endl;
    dk = xk - xk_1;
	// std::cout << "Distance dk:" << std::endl << xk_1(5 + num_vertices) << std::endl;
}

void TrianglesMapping::verify_flips(Triangles& map,
				std::vector<int>& ind_flip) {
	ind_flip.resize(0);
	for (auto f : map.iter_facets()) {
		Eigen::MatrixXd T2_Homo(3,3);
		T2_Homo.col(0) << f.vertex(0).pos()[0], f.vertex(0).pos()[2], 1;
		T2_Homo.col(1) << f.vertex(1).pos()[0], f.vertex(1).pos()[2], 1;
		T2_Homo.col(2) << f.vertex(2).pos()[0], f.vertex(2).pos()[2], 1;
		double det = T2_Homo.determinant();
		assert (det == det);
		if (det < 0) {
		// cout << "flip at face #" << i << " det = " << T2_Homo.determinant() << endl;
		ind_flip.push_back(int(f));
		}
	}
}

int TrianglesMapping::flipsCount(Triangles& map) {
    std::vector<int> ind_flip;
    verify_flips(map, ind_flip);
    return ind_flip.size();
}

void TrianglesMapping::updateUV(Triangles& map, const Eigen::VectorXd& xk) {
    for (auto f : map.iter_facets()) {
        for (int j = 0; j < 3; ++j) {
            int v_ind = int(f.vertex(j));
            f.vertex(j).pos()[0] = xk(v_ind);
            f.vertex(j).pos()[1] = xk(v_ind + num_vertices);
        }
    }
}

// This function determines the maximum step size (alphaMax) that does not cause flips in the mesh.
// It starts with a large alphaMax and iteratively decreases it until no flips are detected.
double TrianglesMapping::determineAlphaMax(const Eigen::VectorXd& xk, const Eigen::VectorXd& dk,
											Triangles& map) {
	double alphaMax = 1.0;
	double decrement = 1e-4;
	std::vector<int> ind_flip;

	while (alphaMax > 0) {
		Eigen::VectorXd xk_new = xk + alphaMax * dk;
        

		for (auto v : map.iter_vertices()) {
			v.pos()[0] = xk_new(int(v));
			v.pos()[2] = xk_new(int(v) + num_vertices);
		}
        updateUV(map, xk_new);

		verify_flips(map, ind_flip);
		if (ind_flip.empty()) {
			break;
		} else {
			alphaMax -= decrement;
		}
	}
	return alphaMax;
}

 double TrianglesMapping::minimum_step_singularities(Triangles& map, Eigen::VectorXd& current, Eigen::VectorXd& distance) {
    double maximum_step = INFINITY;
    // updateUV(map, x);
    for (auto f : map.iter_facets()) {
        // Get quadratic coefficients (ax^2 + b^x + c)
        /*#define U11 f.vertex(0).pos()[0]
        #define U12 f.vertex(0).pos()[2]
        #define U21 f.vertex(1).pos()[0]
        #define U22 f.vertex(1).pos()[2]
        #define U31 f.vertex(2).pos()[0]
        #define U32 f.vertex(2).pos()[2]*/
        #define U11 current(int(f.vertex(0)))
        #define U12 current(int(f.vertex(0)) + num_vertices)
        #define U21 current(int(f.vertex(1)))
        #define U22 current(int(f.vertex(1)) + num_vertices)
        #define U31 current(int(f.vertex(2)))
        #define U32 current(int(f.vertex(2)) + num_vertices)

        #define V11 distance(int(f.vertex(0)))
        #define V12 distance(int(f.vertex(0)) + num_vertices)
        #define V21 distance(int(f.vertex(1)))
        #define V22 distance(int(f.vertex(1)) + num_vertices)
        #define V31 distance(int(f.vertex(2)))
        #define V32 distance(int(f.vertex(2)) + num_vertices)
        
        
        double a = V11*V22 - V12*V21 - V11*V32 + V12*V31 + V21*V32 - V22*V31;
        double b = U11*V22 - U12*V21 - U21*V12 + U22*V11 - U11*V32 + U12*V31 + U31*V12 - U32*V11 + U21*V32 - U22*V31 - U31*V22 + U32*V21;
        double c = U11*U22 - U12*U21 - U11*U32 + U12*U31 + U21*U32 - U22*U31;

        double minimum_positive_root = smallest_position_quadratic_zero(a, b, c);
        maximum_step = std::min(maximum_step, minimum_positive_root);
    }

    return maximum_step;
 }

 /*double TrianglesMapping::minimum_position_root(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& d, int f) {
/*
      Symbolic matlab for equation 4 at the paper (this is how to recreate the formulas below)
      U11 = sym('U11');
      U12 = sym('U12');
      U21 = sym('U21');
      U22 = sym('U22');
      U31 = sym('U31');
      U32 = sym('U32');

      V11 = sym('V11');
      V12 = sym('V12');
      V21 = sym('V21');
      V22 = sym('V22');
      V31 = sym('V31');
      V32 = sym('V32');

      t = sym('t');

      U1 = [U11,U12];
      U2 = [U21,U22];
      U3 = [U31,U32];

      V1 = [V11,V12];
      V2 = [V21,V22];
      V3 = [V31,V32];

      A = [(U2+V2*t) - (U1+ V1*t)];
      B = [(U3+V3*t) - (U1+ V1*t)];
      C = [A;B];

      solve(det(C), t);
      cf = coeffs(det(C),t); % Now cf(1),cf(2),cf(3) holds the coefficients for the polynom. at order c,b,a
}*/

double TrianglesMapping::smallest_position_quadratic_zero(double a, double b, double c) {
    double x1, x2;
    if (a != 0) {
        double delta = pow(b, 2) - 4*a*c;
        if (delta < 0) {
            return INFINITY;
        }
        delta = sqrt(delta);
        x1 = (-b + delta)/ (2*a);
        x2 = (-b - delta)/ (2*a);
    } else {
        x1 = x2 = -b/c;
    }
    assert(std::isfinite(x1));
    assert(std::isfinite(x2));

    double temp = std::min(x1, x2);
    x1 = std::max(x1, x2); x2 = temp;
    if (x1 == x2) {
        return INFINITY; // Means the orientation flips twice, so does it flip?
    }
    // Return the smallest negative root if it exists, otherwise return infinity
    if (x1 > 0) {
        if (x2 > 0) {
            return x2;
        } else {
            return x1;
        }
    } else {
        return INFINITY;
    }
}

void TrianglesMapping::add_energies_jacobians(double& energy_sum, bool flips_linesearch) {
	// schaeffer_e = log_e = conf_e = amips = 0;
	energy_sum = 0;
    distortion_energy.clear();
	for (int i = 0; i < num_triangles; i++) {
        Eigen::Matrix<double, 2, 2> ji, ri, ti, ui, vi;
        Eigen::Matrix<double, 2, 1> sing;
        double mini_energy = 0;

        ji(0, 0) = Ji(i, 0);
        ji(0, 1) = Ji(i, 1);
        ji(1, 0) = Ji(i, 2);
        ji(1, 1) = Ji(i, 3);

        igl::polar_svd(ji, ri, ti, ui, sing, vi);
        double s1 = sing(0);
        double s2 = sing(1);

		if (flips_linesearch) {
            if (strcmp(energy, "ARAP") == 0) {
                mini_energy = Af(i, i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2));
                energy_sum += mini_energy;
            }
            else if (strcmp(energy, "SYMMETRIC-DIRICHLET") == 0) {
                mini_energy = Af(i, i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
                energy_sum += mini_energy;
            }
            else if (strcmp(energy, "EXPONENTIAL-SYMMETRIC-DIRICHLET") == 0) {
                mini_energy = Af(i, i) * exp(exponential_factor * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2)));
                energy_sum += mini_energy;
            }
            else if (strcmp(energy, "HENCKY-STRAIN") == 0) {
                mini_energy = Af(i, i) * (pow(log(s1), 2) + pow(log(s2), 2));
                energy_sum += mini_energy;
            }
            else if (strcmp(energy, "AMIPS") == 0) {
                //
            }
            else if (strcmp(energy, "CONFORMAL-AMIPS-2D") == 0) {
                //
            }
            else if (strcmp(energy, "CONFORMAL-AMIPS-3D") == 0) {
                //
            }
            else if (strcmp(energy, "UNTANGLE-2D") == 0) {
                //
            }
			// schaeffer_energy += Af(i, i) * (pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2));
			// logarithm_energy += Af(i, i) * (pow(log(s1),2) + pow(log(s2),2));
			// double sigma_geo_avg = sqrt(s1*s2);
			// conformal_energy += Af(i, i) * (pow(log(s1/sigma_geo_avg),2) + pow(log(s2/sigma_geo_avg),2));
			// conformal_energy += Af(i, i) * ( (pow(s1,2)+pow(s2,2))/(2*s1*s2) );
			// norm_arap_energy += Af(i, i) * (pow(s1-1, 2) + pow(s2-1, 2));
			// amips +=  Af(i, i) * exp(exponential_factor* (  0.5*( (s1/s2) +(s2/s1) ) + 0.25*( (s1*s2) + (1./(s1*s2)) )  ) );
			// exponential_symmetricDirichlet += Af(i, i) * exp(exponential_factor*(pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2)));
			// amips +=  Af(i, i) * exp(  0.5*( (s1/s2) +(s2/s1) ) + 0.25*( (s1*s2) + (1./(s1*s2)) )  ) ;
		} else {
			if (ui.determinant() * vi.determinant() > 0) {
			energy_sum += Af(i, i) * (pow(s1-1,2) + pow(s2-1,2));
			} else {
			// it is the distance form the flipped thing, this is slow, usefull only for debugging normal arap
			vi.col(1) *= -1;
			energy_sum += Af(i, i) * (Jac[i]-ui*vi.transpose()).squaredNorm();
			}
		}
        // std::cout << "mini_energy: " << Af(i, i) << std::endl;

        distortion_energy.push_back(mini_energy);
	}
}

void TrianglesMapping::compute_energy_gradient(Eigen::VectorXd& grad, bool flips_linesearch, Triangles& map) {
    grad = Eigen::VectorXd::Zero(2 * num_vertices);

    for (int i = 0; i < num_triangles; i++) {
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(Jac[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d Ui = svd.matrixU();
        Eigen::Matrix2d Vi = svd.matrixV();
        Eigen::Vector2d singu = svd.singularValues();

        double s1 = singu(0);
        double s2 = singu(1);

        Eigen::Matrix2d R_i = Ui * Vi.transpose();

        // Compute the gradient of the ARAP energy
        for (int j = 0; j < 3; ++j) {
            Surface::Facet facet_i(map, i);
            int v_ind = int(facet_i.vertex(j));

            Eigen::Vector2d pos;
            pos(0) = facet_i.vertex(j).pos()[0];
            pos(1) = facet_i.vertex(j).pos()[1];
            
            Eigen::Matrix2d grad_J = Eigen::Matrix2d::Zero();
            Eigen::Vector2d e1;
            Eigen::Vector2d e2;
            e1(0) = facet_i.vertex(1).pos()[0] - facet_i.vertex(0).pos()[0];
            e1(1) = facet_i.vertex(1).pos()[1] - facet_i.vertex(0).pos()[1];
            e2(0) = facet_i.vertex(2).pos()[0] - facet_i.vertex(0).pos()[0];
            e2(1) = facet_i.vertex(2).pos()[1] - facet_i.vertex(0).pos()[1];

            // Partial derivatives of e1 and e2 with respect to vertex positions
            Eigen::Matrix2d de1_dv0 = -Eigen::Matrix2d::Identity(); // Partial derivative of e1 w.r.t v0
            Eigen::Matrix2d de1_dv1 = Eigen::Matrix2d::Identity();  // Partial derivative of e1 w.r.t v1
            Eigen::Matrix2d de2_dv0 = -Eigen::Matrix2d::Identity(); // Partial derivative of e2 w.r.t v0
            Eigen::Matrix2d de2_dv2 = Eigen::Matrix2d::Identity();  // Partial derivative of e2 w.r.t v2

            grad_J += (de1_dv0 + de2_dv0); // Combine contributions from all vertices

            Eigen::Matrix2d dJ = grad_J * (Jac[i] - R_i);

            /*if (strcmp(energy, "ARAP") == 0) {
                energy_sum += Af(i, i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2));
            }
            else if (strcmp(energy, "SYMMETRIC-DIRICHLET") == 0) {
                energy_sum += Af(i, i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
            }
            else if (strcmp(energy, "EXPONENTIAL-SYMMETRIC-DIRICHLET") == 0) {
                energy_sum += Af(i, i) * exp(s.exp_factor * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2)));
            }
            else if (strcmp(energy, "HENCKY-STRAIN") == 0) {
                energy_sum += Af(i, i) * (pow(log(s1), 2) + pow(log(s2), 2));
            }
            else if (strcmp(energy, "AMIPS") == 0) {
                //
            }
            else if (strcmp(energy, "CONFORMAL-AMIPS-2D") == 0) {
                //
            }
            else if (strcmp(energy, "CONFORMAL-AMIPS-3D") == 0) {
                //
            }
            else if (strcmp(energy, "UNTANGLE-2D") == 0) {
                //
            }*/

            for (int k = 0; k < 2; ++k) {
                grad(v_ind + k * num_vertices) += 2 * Af(i, i) * dJ(k, 0) * (s1 - 1) + dJ(k, 1) * (s2 - 1);
            }

            if (flips_linesearch && Ui.determinant() * Vi.determinant() <= 0) {
                Vi.col(1) *= -1;
                Eigen::Matrix2d flip_R_i = Ui * Vi.transpose();
                Eigen::Matrix2d dJ_flipped = grad_J * (Jac[i] - flip_R_i);

                for (int k = 0; k < 2; ++k) {
                    grad(v_ind + k * num_vertices) += Af(i, i) * (dJ_flipped(k, 0) * (s1 - 1) + dJ_flipped(k, 1) * (s2 - 1));
                }
            }
        }
    }
}

void TrianglesMapping::computeGradient(Eigen::VectorXd& x, Eigen::VectorXd& grad, Triangles& map) {
    double h = 1e-5; // small step for finite differences
    double original_energy;
    
    // Compute the original energy at x
    /*for (auto v : map.iter_vertices()) {
        v.pos()[0] = x(int(v));
        v.pos()[2] = x(int(v) + num_vertices);
    }*/
    updateUV(map, x);
    jacobian_rotation_area(map, true);
    add_energies_jacobians(original_energy, true);

    // Iterate over each dimension of x to compute the partial derivative
    for (int i = 0; i < x.size(); ++i) {
        Eigen::VectorXd x_plus_h = x;
        x_plus_h[i] += h;

        double new_energy;
        /*for (auto v : map.iter_vertices()) {
            v.pos()[0] = x_plus_h(int(v));
            v.pos()[2] = x_plus_h(int(v) + num_vertices);
        }*/
        updateUV(map, x_plus_h);
        jacobian_rotation_area(map, true);
        add_energies_jacobians(new_energy, true);

        grad[i] = (new_energy - original_energy) / h;
        std::cout << i << std::endl;
    }
}

void TrianglesMapping::computeAnalyticalGradient(Eigen::VectorXd& x, Eigen::VectorXd& grad, Triangles& map) {
    // Ensure grad is the correct size
    grad.resize(x.size());
    grad.setZero();
    
    // Reset vertex positions according to x
    /*for (auto v : map.iter_vertices()) {
        v.pos()[0] = x(int(v));
        v.pos()[2] = x(int(v) + num_vertices);
    }*/
    updateUV(map, x);
    
    // Compute the energy and its gradient (Jacobian)
    jacobian_rotation_area(map, true); // This function should set the internal state needed for gradients

    // Accumulate the gradients from Dx
    for (int k = 0; k < Dx.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Dx, k); it; ++it) {
            grad[it.row()] += Af(k, k) * it.value();
        }
    }

    // Accumulate the gradients from Dy, offset by num_vertices for the y-component
    for (int k = 0; k < Dy.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Dy, k); it; ++it) {
            grad[it.row() + num_vertices] += Af(k, k) * it.value();
        }
    }

    /*double s1_g = 2* (s1-pow(s1,-3)); 
          double s2_g = 2 * (s2-pow(s2,-3));
          m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1)));*/
}

double TrianglesMapping::lineSearch(Eigen::VectorXd& xk_current, Eigen::VectorXd& dk, Triangles& map) {
    // Line search using Wolfe conditions
    double c1 = 1e-4; // 1e-5
    double c2 = 0.9; // 0.99
    std::cout << "lineSearch: " << std::endl;
    double alphaMax = minimum_step_singularities(map, xk_current, dk);
    // double alphaMax = determineAlphaMax(xk_current, dk, map);
    
    double alphaStep = 0.99 * alphaMax;
    alphaStep = std::min(1.0, 0.8 * alphaMax);
    double alphaTest = 0.99 * alphaMax;
    double alphaStep2 = std::min(1.0, 0.8 * alphaMax);

    Eigen::VectorXd pk = xk_current + alphaStep * dk;
    std::cout << "alphaStep: " << alphaStep << std::endl;

    updateUV(map, xk_current);
    jacobian_rotation_area(map, true);
    double ener, new_ener;
    add_energies_jacobians(ener, true);
    
    // Compute gradient of xk_current
    Eigen::VectorXd grad_xk = Eigen::VectorXd::Zero(xk_current.size());
    // computeGradient(xk_current, grad_xk, map);
	// computeAnalyticalGradient(xk_current, grad_xk, map);
    compute_energy_gradient(grad_xk, true, map);
    
    /*for (auto v : map.iter_vertices()) {
        v.pos()[0] = pk(int(v));
        v.pos()[2] = pk(int(v) + num_vertices);
    }*/
    Eigen::VectorXd grad_pk = Eigen::VectorXd::Zero(pk.size());

    // Wolfe conditions
    auto wolfe1 = [&]() {
        return new_ener > ener + c1 * alphaStep * grad_xk.dot(dk);
    };

    auto wolfe2 = [&]() {
        return abs(grad_pk.dot(dk)) > c2 * abs(grad_xk.dot(dk));
    };

    // // Initial check
    // if (wolfe1() && wolfe2()) {
    //     return alpha;
    // }

    // Bisection line search with max_iter limit
    double alphaLow = 0.0;
    double alphaHigh = alphaStep;
    int max_iter = 5;
    int iter = 0;

    /*while (iter < max_iter) {
        if (wolfe1()) {
            alphaHigh = alphaStep;
            alphaStep = (alphaLow + alphaHigh) / 2.0;
            
        } else if (wolfe2()) {
            updateUV(map, pk);
            jacobian_rotation_area(map, true);
            add_energies_jacobians(new_ener, true);
            // Compute gradient of pk
            
            // computeGradient(pk, grad_pk, map);
            // computeAnalyticalGradient(pk, grad_pk, map);
            compute_energy_gradient(grad_pk, true, map);

            if (grad_pk.dot(dk) > 0) {
                alphaHigh = alphaStep;
            }
            else {
                alphaLow = alphaStep;
            }
            alphaStep = (alphaLow + alphaHigh) / 2.0;
        } else {
            break;
        }

        pk = xk_current + alphaStep * dk;

        // Update pk positions
        /*for (auto v : map.iter_vertices()) {
            v.pos()[0] = pk(int(v));
            v.pos()[2] = pk(int(v) + num_vertices);
        }///////
        updateUV(map, pk);
        jacobian_rotation_area(map, true);
        add_energies_jacobians(new_ener, true);

        // Compute gradient of pk
        // computeGradient(pk, grad_pk, map);
		// computeAnalyticalGradient(pk, grad_pk, map);
        compute_energy_gradient(grad_pk, true, map);

        iter++;
    }*/


    while (iter < max_iter) {
        if (new_ener > ener) {
            alphaHigh = alphaStep2;
            alphaStep2 = (alphaLow + alphaHigh) / 2.0;
            
        } else {
            alphaLow = alphaStep2;
            alphaStep2 = (alphaLow + alphaHigh) / 2.0;
        }

        pk = xk_current + alphaStep2 * dk;

        // Update pk positions
        /*for (auto v : map.iter_vertices()) {
            v.pos()[0] = pk(int(v));
            v.pos()[2] = pk(int(v) + num_vertices);
        }*/
        updateUV(map, pk);
        jacobian_rotation_area(map, true);
        add_energies_jacobians(new_ener, true);

        // Compute gradient of pk
        // computeGradient(pk, grad_pk, map);
		// computeAnalyticalGradient(pk, grad_pk, map);
        compute_energy_gradient(grad_pk, true, map);

        iter++;
    }


    return alphaStep2;
}

void TrianglesMapping::nextStep(Triangles& map) {
	// Perform line search to find step size alpha
	double alpha = lineSearch(xk_1, dk, map);

	// Update the solution xk
	xk = xk_1 + alpha * dk;

	// std::cout << "The new solution is:\n" << xk << std::endl;
	std::cout << "Step size alpha: " << alpha << std::endl;

	/*for (auto v : mLocGlo.iter_vertices()) {
		v.pos()[0] = xk(int(v));
		v.pos()[2] = xk(int(v) + num_vertices);
	}*/
    updateUV(map, xk);
}

void TrianglesMapping::bound_vertices_circle_normalized(Triangles& map) {
    double area_b = 0;
    for (auto f : map.iter_facets()) {
        // area_b += calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
        area_b += unsigned_area(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
    }
    double radius = sqrt(area_b / (M_PI));

    int total_len = bound_sorted.size();
    int len_i = 0;
    for (auto v : bound_sorted) {
        double frac = len_i * (2. * M_PI) / total_len;
        map.points[v][0] = radius * cos(frac);
        map.points[v][1] = radius * sin(frac);
        len_i++;
    }
}

void TrianglesMapping::Tut63(const char* name, int weights) {
    std::filesystem::path filepath = name;
    std::string filepath_str_ext = filepath.extension().string();
    std::string filepath_str_stem = filepath.stem().string();
    const char* ext = filepath_str_ext.c_str();
    const char* stem = filepath_str_stem.c_str();

    char ext2[12] = ".geogram";
    char method[20] = "_barycentre";
    char weight1[20] = "_uniform";
    char weight2[20] = "_cotan";
    // char attribute[20] = "_distortion";

    // Create directory if it doesn't exist
    std::filesystem::path stem_dir(stem);
    if (!std::filesystem::exists(stem_dir)) {
        std::filesystem::create_directory(stem_dir);
    }

    strcpy(output_name_geo, stem);
    strcat(output_name_geo, "/");
    strcat(output_name_geo, energy);
    strcat(output_name_geo, "/");
    strcat(output_name_geo, stem);
    strcat(output_name_geo, method);
    if (weights == 1) {
        strcat(output_name_geo, weight1);
    } else if (weights == 2) {
        strcat(output_name_geo, weight2);
    }
    strcpy(output_name_obj, output_name_geo);
    // strcat(output_name, attribute);
    strcat(output_name_geo, ext2);
    strcat(output_name_obj, ".obj");

    strcpy(times_txt, stem);
    strcat(times_txt, "/");
    strcat(times_txt, energy);
    strcat(times_txt, "/");
    strcat(times_txt, stem);
    strcat(times_txt, ".txt");
    std::ofstream timeFile(times_txt); // Open a file for writing times
    auto start = std::chrono::high_resolution_clock::now();
    totalStart = std::chrono::high_resolution_clock::now(); // Start timing for total duration
    totalTime = 0; // Initialize total time accumulator

    read_by_extension(name, mTut);
    read_by_extension(name, mOri);

    double maxH = mTut.points[0][1];
    double minH = mTut.points[0][1];
    int nverts = mTut.nverts();
    for (int i = 0; i < mTut.nverts(); i++) {
        if (mTut.points[i][1] < minH) {
            minH = mTut.points[i][1];
        } else if (mTut.points[i][1] > maxH) {
            maxH = mTut.points[i][1];
        }
    }
    double seaLevel = 0.;
    double margin = 1.0 / nverts * 100;
    double dcuttingSurface = (maxH - minH) * margin;

    DBOUT("The number of vertices is: " << mTut.nverts() << ", facets: " << mTut.nfacets() << ", corners: " << mTut.ncorners() << std::endl);

    mOri.connect();
    mTut.connect();

    // reference_mesh(mOri);

    int ind = 0;
    Af = Eigen::MatrixXd::Zero(mOri.nfacets(), mOri.nfacets());
    for (auto f : mOri.iter_facets()) {
        // Af(ind, ind) = std::sqrt(calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()));
        // Af(ind, ind) = area[int(f)];
        Af(ind, ind) = unsigned_area(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
        ind++;
    }

    igl::read_triangle_mesh(name, V, F);

    /*int fixed = 0;
    Eigen::VectorXd x_B_ = Eigen::VectorXd::Zero(nverts);
    Eigen::VectorXd x_I_ = Eigen::VectorXd::Zero(nverts);
    int insider = 0;*/
    std::unordered_map<int, std::vector<int>> neighbor;
    std::unordered_set<int> bound;
    std::set<int> bound_halfedges;

    for (auto he : mOri.iter_halfedges()) {
        if (!he.opposite().active()) {
            neighbor[he.from()].push_back(he.to());
            neighbor[he.to()].push_back(he.from());
            bound.insert(he.from());
            bound.insert(he.to());

            bound_halfedges.insert(he);
        }
    }

    // std::vector<int> bound_sorted;
    int missing = 0;
    if (!bound.empty()) {
        std::unordered_set<int> visited;
        int start = *bound.begin();
        bound_sorted.push_back(start);
        visited.insert(start);

        while (bound_sorted.size() < bound.size()) {
            int last = bound_sorted.back();
            bool found = false;
            
            for (int next : neighbor[last]) {
                if (visited.find(next) == visited.end()) {
                    bound_sorted.push_back(next);
                    visited.insert(next);
                    found = true;
                    break;
                }
            }

            if (!found) {
                missing++;
                break;
            }
        }
        std::cout << "Number of miss : " << missing << std::endl;
    }

    for (const auto& pair : neighbor) {
        std::cout << "Key: " << pair.first << " Values: ";
        for (const int& val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << bound.size() << "  " << bound_sorted.size() << std::endl;

    /*bound_sorted.clear();

    int init = *bound_halfedges.begin();
    std::vector<int> bound_sorted2;
    missing = 0;
    if (!bound.empty()) {
        Surface::Halfedge h(mOri, init);
        int start = h.from();
        bound_sorted.push_back(start);

        while (bound_sorted.size() < bound.size()) {
            bool found = false;
            Surface::Halfedge h_next1 = h.next();
            if (h_next1.active()) {
                if (bound_halfedges.contains(h_next1)) {
                    h = h_next1;
                    bound_sorted.push_back(h.from());
                    found = true;
                    continue;
                }
            }
            Surface::Halfedge h_next2 = h.next().opposite().next();
            if (h_next2.active()) {
                if (bound_halfedges.contains(h_next2)) {
                    h = h_next2;
                    bound_sorted.push_back(h.from());
                    found = true;
                    continue;
                }
            }
            Surface::Halfedge h_next3 = h.next().opposite().next().opposite().next();
            if (h_next3.active()) {
                if (bound_halfedges.contains(h_next3)) {
                    h = h_next3;
                    bound_sorted.push_back(h.from());
                    found = true;
                    continue;
                }
            }

            Surface::Halfedge h_next4 = h.next().opposite().next().opposite().next().opposite().next();
            if (h_next4.active()) {
                if (bound_halfedges.contains(h_next4)) {
                    h = h_next4;
                    bound_sorted.push_back(h.from());
                    found = true;
                    continue;
                }
            }

            if (!found) {
                missing++;
                break;
            }
        }
        std::cout << "Number of miss : " << missing << std::endl;
    }*/

    std::cout << bound.size() << "  " << bound_sorted.size() << std::endl;

    // exit(EXIT_FAILURE);

    for (int v : bound_sorted) {
        std::cout << v << " " << std::endl;
    }
    // exit(EXIT_FAILURE);
    
    /*std::map<int, int> valueCounts;
    for (const auto& pair : neighbor) {
        valueCounts[pair.second]++;
    }
    for (const auto& pair : valueCounts) {
        if (pair.second > 2) {
            std::cerr << "Error: A bound vertex has more than two neighbors." << std::endl;
            exit(EXIT_FAILURE);
        }
    }*/
    
    int fixed = 0;
    // std::set<int> blade;
    Eigen::VectorXd x_B_ = Eigen::VectorXd::Zero(nverts);
    for (int i = 0; i < mTut.nverts(); i++) {
        Surface::Vertex vi = Surface::Vertex(mOri, i);
        if (bound.contains(vi)) {
            blade.insert(vi);
            x_B_(i) = fixed;
            fixed++;
        }
    }

    Eigen::VectorXd x_I_ = Eigen::VectorXd::Zero(nverts);
    int insider = 0;
    std::set<int> plane;
    for (int i = 0; i < mTut.nverts(); i++) {
        Surface::Vertex vi = Surface::Vertex(mOri, i);
        if (!bound.contains(vi)) {
            plane.insert(vi);
            x_I_(i) = insider;
            insider++;
        }
    }

    Eigen::SparseMatrix<double> A_II(nverts - fixed, nverts - fixed);
    Eigen::SparseMatrix<double> A_IB(nverts - fixed, fixed);
    Eigen::VectorXd b_I = Eigen::VectorXd::Zero(nverts - fixed);
    Eigen::MatrixXd x_B = Eigen::MatrixXd::Zero(fixed, 2);

    Eigen::SparseMatrix<double> A_II_A_BB(nverts, nverts);
    for (int i = 0; i < fixed; ++i) {
        A_II_A_BB.insert(i, i) = 1;
    }

    Eigen::MatrixXd lhsF = Eigen::MatrixXd::Zero(nverts, 2);

    bound_vertices_circle_normalized(mTut);

    int vert = 0;
    int bb = 0;
    double progress = 0;
    for (int i = 0; i < mTut.nverts(); i++) {
        Surface::Vertex vi = Surface::Vertex(mOri, i);
        if (bound.contains(vi)) {
            x_B(bb, 0) = mTut.points[i][0];
            x_B(bb, 1) = mTut.points[i][1];

            lhsF(bb, 0) = mTut.points[i][0];
            lhsF(bb, 1) = mTut.points[i][1];

            bb++;
        } else {
            Surface::Halfedge depart = Surface::Vertex(mOri, i).halfedge();
            Surface::Halfedge variable = depart;
            double count = 0;
            std::vector<int> neighbors;
            if (weights == 1) {
                std::map<int, double> cotan;
                if (depart.opposite().active())
                    variable = variable.opposite().next();
                neighbors.push_back(depart.to());
                cotan.insert(std::make_pair(neighbors.back(), 1));
                count += 1;
                while (depart != variable && variable.active()) {
                    neighbors.push_back(variable.to());
                    cotan.insert(std::make_pair(neighbors.back(), 1));
                    count += 1;
                    if (!variable.opposite().active())
                        break;
                    variable = variable.opposite().next();
                }

                int ree = x_I_(i);
                A_II.insert(ree, ree) = -count;
                A_II_A_BB.insert(ree + fixed, ree + fixed) = -count;

                for (auto const& [key, val] : cotan) {
                    if (blade.contains(key)) {
                        int re_ne2 = x_B_(key);
                        A_IB.insert(ree, re_ne2) = val;
                        A_II_A_BB.insert(ree + fixed, re_ne2) = val;
                    } else {
                        int re_ne = x_I_(key);
                        A_II.insert(ree, re_ne) = val;
                        A_II_A_BB.insert(ree + fixed, re_ne + fixed) = val;
                    }
                }
            } else if (weights == 2) {
                std::map<int, double> cotan;
                if (depart.opposite().active())
                    variable = variable.opposite().next();

                double cotan_alpha = calculateCotan(depart.next().from().pos(), depart.next().to().pos(), depart.prev().to().pos(), depart.prev().from().pos());
                double cotan_beta = calculateCotan(depart.opposite().prev().to().pos(), depart.opposite().prev().from().pos(), depart.opposite().next().from().pos(), depart.opposite().next().to().pos());
                double cotan_gamma = calculateCotan(depart.next().to().pos(), depart.next().from().pos(), depart.from().pos(), depart.to().pos());

                double w_ij = cotan_alpha + cotan_beta;
                double voronoi = 0.125 * (cotan_gamma * (depart.from().pos() - depart.to().pos()).norm2() + cotan_alpha * (depart.prev().to().pos() - depart.prev().from().pos()).norm2());

                neighbors.push_back(depart.to());
                cotan.insert(std::make_pair(neighbors.back(), w_ij));
                count += w_ij;

                while (depart != variable && variable.active()) {
                    cotan_alpha = calculateCotan(variable.next().from().pos(), variable.next().to().pos(), variable.prev().to().pos(), variable.prev().from().pos());
                    cotan_beta = calculateCotan(variable.opposite().prev().to().pos(), variable.opposite().prev().from().pos(), variable.opposite().next().from().pos(), variable.opposite().next().to().pos());
                    cotan_gamma = calculateCotan(variable.next().to().pos(), variable.next().from().pos(), variable.from().pos(), variable.to().pos());

                    w_ij = cotan_alpha + cotan_beta;
                    voronoi += 0.125 * (cotan_gamma * (variable.from().pos() - variable.to().pos()).norm2() + cotan_alpha * (variable.prev().to().pos() - variable.prev().from().pos()).norm2());

                    neighbors.push_back(variable.to());
                    cotan.insert(std::make_pair(neighbors.back(), w_ij));
                    count += w_ij;
                    if (!variable.opposite().active())
                        break;
                    variable = variable.opposite().next();
                }

                int ree = x_I_(i);
                A_II.insert(ree, ree) = -count / (2 * voronoi);
                A_II_A_BB.insert(ree + fixed, ree + fixed) = -count / (2 * voronoi);

                for (auto const& [key, val] : cotan) {
                    if (blade.contains(key)) {
                        int re_ne2 = x_B_(key);
                        A_IB.insert(ree, re_ne2) = val / (2 * voronoi);
                        A_II_A_BB.insert(ree + fixed, re_ne2) = val / (2 * voronoi);
                    } else {
                        int re_ne = x_I_(key);
                        A_II.insert(ree, re_ne) = val / (2 * voronoi);
                        A_II_A_BB.insert(ree + fixed, re_ne + fixed) = val / (2 * voronoi);
                    }
                }
            }
        }
        vert++;
        progress = std::round(static_cast<float>(vert) / nverts * 100000.0f) / 100000.0f * 100;
        DBOUT("Vertex " << vert << "/" << nverts << " (" << progress << " %) --- dim1: " << mOri.points[i][0] << ", dim2: " << mOri.points[i][1] << ", dim3: " << mOri.points[i][2] << std::endl);
    }
    
    // !! A DEBOGUER !!
    /*Eigen::VectorXd lhs = b_I - A_IB * x_B;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_II);
    if (solver.info() != Eigen::Success) {
        // Decomposition failed
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }
    Eigen::VectorXd x_I = solver.solve(lhs);
    if (solver.info() != Eigen::Success) {
        // Solving failed
        std::cerr << "Solving failed" << std::endl;
        return;
    }*/

    // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    // Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> solver;
   /*Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    // Regularization (if necessary)
    double regularizationTerm = 1e-4; // Small positive value
    for (int i = 0; i < A_II_A_BB.rows(); ++i) {
        A_II_A_BB.coeffRef(i, i) += regularizationTerm;
    }

    solver.compute(A_II_A_BB);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }

    Eigen::VectorXd x_I_full = solver.solve(lhsF);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed" << std::endl;
        return;
    }*/

    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    // solver.compute(A_II_A_BB);

    A_II_A_BB.makeCompressed();
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(A_II_A_BB);
    solver.factorize(A_II_A_BB);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }
    
    Eigen::MatrixXd x_I_full = solver.solve(lhsF);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed" << std::endl;
        return;
    }

    // std::cout << x_I_full << std::endl;
    EigenMap = x_I_full;

    for (int plan : plane) {
        int re = x_I_(plan);
        // mTut.points[plan][0] = x_I(re, 0);
	    // mTut.points[plan][2] = x_I(re, 1);

        mTut.points[plan][0] = x_I_full(re + fixed, 0);
        mTut.points[plan][2] = seaLevel;
        mTut.points[plan][1] = x_I_full(re + fixed, 1);
    }
    for (int blad : blade) {
        mTut.points[blad][2] = seaLevel;
    }

    // for (int plan : plane) {
    //     int re = x_I_(plan);
    //     mTut.points[plan][0] = x_I_full(re + fixed, 0);
    //     mTut.points[plan][1] = x_I_full(re + fixed, 2);
    //     mTut.points[plan][2] = x_I_full(re + fixed, 1);
    // }
    // for (int blad : blade) {
    //     int re = x_B_(blad);
    //     mTut.points[blad][1] = x_I_full(re, 2);
    // }

    FacetAttribute<double> fa2(mOri);
    for (auto f : mOri.iter_facets()) {
        // double area = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
        double area = unsigned_area(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
        fa2[f] = area;
        fOriMap[int(f)] = area;
    }

    CornerAttribute<double> he(mTut);
    for (auto f : mTut.iter_halfedges()) {
        if (blade.contains(f.from()) || blade.contains(f.to())) {
            he[f] = 404;
        } else {
            he[f] = 0;
        }
    }

    Surface::Facet f(mTut, 0);
    // double minArea = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fOriMap[0];
    double minArea = unsigned_area(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fOriMap[0];
    double maxArea = minArea;
    FacetAttribute<double> fa(mTut);
    for (auto f : mTut.iter_facets()) {
        // fa[f] = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fa2[f];
        fa[f] = unsigned_area(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fa2[f];
        if (fa[f] < minArea) {
            minArea = fa[f];
        } else if (fa[f] > maxArea) {
            maxArea = fa[f];
        }
    }

    double minEnergy = 0.0;
    double maxEnergy = 0.0;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    totalTime += duration; // Accumulate total time
    if (timeFile.is_open()) {
        timeFile << "0" << "|" << duration << "|"; // Write iteration number and duration to file
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();
    if (timeFile.is_open()) {
        timeFile << totalDuration << "|"; // Log total time
        timeFile << minArea << "|" << maxArea << "|" << minEnergy << "|" << maxEnergy << "|"; // Log min/max area and distortion
        timeFile << mTut.nverts() << "|" << mTut.nfacets() << "|" << mTut.ncorners() << "\n";
    }

    if (timeFile.is_open()) {
        timeFile.close();
    }

    write_by_extension(output_name_geo, mTut, { {}, {{"AreaRatio", fa.ptr}}, {{"Halfedge", he.ptr}} });
    write_by_extension(output_name_obj, mTut);
    igl::read_triangle_mesh(output_name_obj, V_1, F_1);

    // #ifdef _WIN32
    //     // Open the generated mesh with Graphite
    //     int result = system((getGraphitePath() + " " + output_name_geo).c_str());
    // #endif
    // #ifdef linux
    //     system((std::string("graphite ") + output_name_geo).c_str());
    // #endif
}

void TrianglesMapping::LocalGlobalParametrization(const char* map) {
    read_by_extension(map, mLocGlo);
    std::cout << "Yo long time no see\n" << max_iterations << std::endl;
    mLocGlo.connect();

    std::filesystem::path filepath = map;
    std::string filepath_str_ext = filepath.extension().string();
    std::string filepath_str_stem = filepath.stem().string();
    const char* ext = filepath_str_ext.c_str();
    const char* stem = filepath_str_stem.c_str();
    const char* first_space = strchr(stem, '_'); // Find the first space in stem
    size_t first_word_length = first_space ? (size_t)(first_space - stem) : strlen(stem); // Calculate length of the first word
    
    // char times_txt[100];
    /*strncpy(times_txt, stem, first_word_length);
    times_txt[first_word_length] = '\0';
    strcat(times_txt, "/");
    strcat(times_txt, energy);
    strcat(times_txt, "/");
    strncat(times_txt, stem, first_word_length);
    strcat(times_txt, ".txt");*/
    std::ofstream timeFile(times_txt, std::ios::app); // Append mode

    char ext2[12] = ".geogram";
    char method[20] = "_local_global_";
    // char attribute[20] = "_distortion_";
    char numStr[20];
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);
    std::cout << orientation << std::endl;
    if (strcmp(orientation, "xyz") == 0) {
        compute_surface_gradient_matrix(V, F, F1, F2, Dx, Dy);
    }
    else if (strcmp(orientation, "xzy") == 0) {
        compute_surface_gradient_matrix(V, F, F1, F3, Dx, Dy);
    }
    else if (strcmp(orientation, "yxz") == 0) {
        compute_surface_gradient_matrix(V, F, F2, F1, Dx, Dy);
    }
    else if (strcmp(orientation, "yzx") == 0) {
        compute_surface_gradient_matrix(V, F, F2, F3, Dx, Dy);
    }
    else if (strcmp(orientation, "zxy") == 0) {
        compute_surface_gradient_matrix(V, F, F3, F1, Dx, Dy);
    }
    else if (strcmp(orientation, "zyx") == 0) {
        compute_surface_gradient_matrix(V, F, F3, F2, Dx, Dy);
    }
	for (int i = 1; i <= max_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        jacobian_rotation_area(mLocGlo, false);
        std::cout << "jacobian_rotation_area(mLocGlo);" << std::endl;
        // update_weights();
        std::cout << "update_weights();" << std::endl;
        least_squares();
        std::cout << "least_squares();" << std::endl;
        nextStep(mLocGlo);
        std::cout << "nextStep(mLocGlo); FIN ITERATION " << i << std::endl;
        
        output_name_geo[0] = '\0'; // Clear output_name
        strncpy(output_name_geo, stem, first_word_length);
        output_name_geo[first_word_length] = '\0'; // Ensure null-termination
        strcat(output_name_geo, "/");
        strcat(output_name_geo, energy);
        strcat(output_name_geo, "/");
        strncat(output_name_geo, stem, first_word_length);

        strcat(output_name_geo, method);
        strcat(output_name_geo, energy);
        strcat(output_name_geo, "_");
        // strcat(output_name, attribute);
        sprintf(numStr, "%d", i);
        strcat(output_name_geo, numStr);
        strcpy(output_name_obj, output_name_geo);
        strcat(output_name_geo, ext2);
        strcat(output_name_obj, ".obj");

        // reference_mesh(mLocGlo);

        CornerAttribute<double> he(mLocGlo);
        for (auto f : mLocGlo.iter_halfedges()) {
            if (blade.contains(f.from()) || blade.contains(f.to())) {
                he[f] = 404;
            } else {
                he[f] = 0;
            }
        }

        Surface::Facet f(mLocGlo, 0);
        // double minArea = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fOriMap[0];
        double minArea = unsigned_area(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fOriMap[0];
        double maxArea = minArea;
        FacetAttribute<double> fa_a(mLocGlo);
        for (auto f : mLocGlo.iter_facets()) {
            // fa_a[f] = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fOriMap[int(f)];
            fa_a[f] = unsigned_area(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fOriMap[int(f)];
            if (fa_a[f] < minArea) {
                minArea = fa_a[f];
            } else if (fa_a[f] > maxArea) {
                maxArea = fa_a[f];
            }
        }

        double minEnergy = distortion_energy[0];
        double maxEnergy = distortion_energy[0];
        FacetAttribute<double> fa(mLocGlo);
        for (auto f : mLocGlo.iter_facets()) {
                fa[f] = distortion_energy[int(f)];
                if (fa[f] < minEnergy) {
                    minEnergy = fa[f];
                } else if (fa[f] > maxEnergy) {
                    maxEnergy = fa[f];
                }
                // std::cout << "distortion_energy[int(f)] : " << distortion_energy[int(f)] << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalTime += duration; // Accumulate total time
        if (timeFile.is_open()) {
            timeFile << i << "|" << duration << "|"; // Write iteration number and duration to file
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();
        if (timeFile.is_open()) {
            timeFile << totalDuration << "|"; // Log total time
            timeFile << minArea << "|" << maxArea << "|" << minEnergy << "|" << maxEnergy << "|"; // Log min/max area and distortion
            timeFile << mLocGlo.nverts() << "|" << mLocGlo.nfacets() << "|" << mLocGlo.ncorners() << "\n";
        }


        write_by_extension(output_name_geo, mLocGlo, { {}, {{"Energy", fa.ptr}, {"AreaRatio", fa_a.ptr}}, {{"Halfedge", he.ptr}} });
        write_by_extension(output_name_obj, mLocGlo);
        igl::read_triangle_mesh(output_name_obj, V_1, F_1);
        std::cout << "write_by_extension(output_name_geo, mLocGlo);" << std::endl;
        // #ifdef _WIN32
        //     int result = system((getGraphitePath() + " " + output_name_geo).c_str());
        // #endif
        // #ifdef linux
        //     system((std::string("graphite ") + output_name_geo).c_str());
        // #endif
	}

    if (timeFile.is_open()) {
        timeFile.close();
    }
}

void updateProgressBar(int progress) {
    const int barWidth = 50; // Width of the progress bar in characters

    std::cout << "[";
    int pos = barWidth * progress / 100;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << progress << " %\r";
    std::cout.flush(); // Important to ensure the output is updated immediately
}

int main(int argc, char** argv) {

    auto start = std::chrono::high_resolution_clock::now();
    TrianglesMapping Init(argc, argv);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
    
	Init.LocalGlobalParametrization(Init.getOutput());

    std::cout << Init.getOutput() << std::endl;

	for (int progress = 0; progress <= 100; ++progress) {
        updateProgressBar(progress);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << std::endl;


	/*


	----------------------------------------3D to 2D----------------------------------------


Export .obj ou .geogram, coordonnées de texture pour graphite


//////////////////////////////////////////////////ENERGY//////////////////////////////////////////////////



void SymmetricDirichlet::compute_negative_gradient(const Eigen::MatrixXd& V,
					  const Eigen::MatrixXi& F,
					  const Eigen::MatrixXd& uv,
					  Eigen::MatrixXd& neg_grad) {

	//cout << "computing gradient the old way" << endl;
  	Eigen::VectorXd dblArea_p;
  	igl::doublearea(uv,F, dblArea_p);
  	//cout << "V area sum = " << m_dblArea_orig.sum() << " and uv area sum = " << dblArea_p.sum() << endl;

	neg_grad.setZero();
	// for DBG
	// Eigen::MatrixXd left_grad(neg_grad.rows(),neg_grad.cols()); left_grad.setZero();
	// Eigen::MatrixXd right_grad(neg_grad.rows(),neg_grad.cols()); right_grad.setZero();
	for (int i = 0; i < F.rows(); i++) {
		// add to the vertices gradient
		
		double energy_left_part = m_cached_l_energy_per_face[i];
		double energy_right_part = m_cached_r_energy_per_face[i];

		double t_orig_area = m_dblArea_orig(i)/2;
		double t_uv_area = dblArea_p(i)/2;
		double left_grad_const = -pow(t_orig_area,2)/pow(t_uv_area,3);

		for (int j = 0; j < 3; j++) {
			int v1 = j; int v2 = (j+1)%3; int v3 = (j+2)%3;
			int v1_i = F(i,j); int v2_i = F(i,(j + 1)%3); int v3_i = F(i,(j + 2)%3);
			// compute left gradient
			Eigen::RowVector2d c_left_grad;
			Eigen::RowVector2d rotated_left_grad = (left_grad_const * (uv.row(v2_i)-uv.row(v3_i)));
			c_left_grad(0) = rotated_left_grad(1); c_left_grad(1) = -rotated_left_grad(0);
			
			// compute right gradient
			Eigen::RowVector2d c_right_grad;
			//− cot(θ2)U3 − cot(θ3)U2 + (cot(θ2) + cot(θ3))U1
			// note: the entries for this function are half of the contangents
			
			c_right_grad = -m_cot_entries(i,v2)*uv.row(v3_i) - m_cot_entries(i,v3)*uv.row(v2_i) 
						+ (m_cot_entries(i,v2) + m_cot_entries(i,v3))*uv.row(v1_i);
			
			// product rule (and subtract from vector cause we compute the negative of the gradient)
			neg_grad.row(v1_i) = neg_grad.row(v1_i) - (c_left_grad * energy_right_part + c_right_grad * energy_left_part);

			// for DBG
			// left_grad.row(v1_i) = left_grad.row(v1_i) + c_left_grad;
			// right_grad.row(v1_i) = right_grad.row(v1_i) + c_right_grad;
		}
	}
	//zero_out_const_vertices_search_direction(neg_grad);
}

double SymmetricDirichlet::compute_energy(const Eigen::MatrixXd& V,
    				 const Eigen::MatrixXi& F,
    				 const Eigen::MatrixXd& uv) {
  precompute(V,F); // in case we need precomputation
	double energy = 0;
	//cout << "normal compute energy!" << endl;

	for (int i = 0; i < F.rows(); i++) {
		double l_part = compute_face_energy_left_part(V,F,uv,i);
		double r_part = compute_face_energy_right_part(V,F,uv,i, m_dblArea_orig(i));

		energy += l_part * r_part;

		// cache results for the gradient use
		m_cached_l_energy_per_face[i] = l_part;
		m_cached_r_energy_per_face[i] = r_part;

	}
	return energy;
}

double SymmetricDirichlet::compute_face_energy_left_part(const Eigen::MatrixXd& V,
    				 const Eigen::MatrixXi& F,
    				 const Eigen::MatrixXd& uv, int f) {
	// compute current triangle squared area
    auto rx = uv(F(f,0),0)-uv(F(f,2),0);
    auto sx = uv(F(f,1),0)-uv(F(f,2),0);
    auto ry = uv(F(f,0),1)-uv(F(f,2),1);
    auto sy = uv(F(f,1),1)-uv(F(f,2),1);
    double dblAd = rx*sy - ry*sx;
	double uv_sqrt_dbl_area = dblAd*dblAd;
    

    return (1 + (m_dbl_sqrt_area(f)/uv_sqrt_dbl_area));
}

double SymmetricDirichlet::compute_face_energy_right_part(const Eigen::MatrixXd& V,
    				 const Eigen::MatrixXi& F,
    				 const Eigen::MatrixXd& uv,int f_idx,
    				 double orig_t_dbl_area) {
	int v_1 = F(f_idx,0); int v_2 = F(f_idx,1); int v_3 = F(f_idx,2);

	double part_1 = (uv.row(v_3)-uv.row(v_1)).squaredNorm() * m_cached_edges_1[f_idx];
	part_1 += (uv.row(v_2)-uv.row(v_1)).squaredNorm()* m_cached_edges_2[f_idx];
	part_1 /= (2*orig_t_dbl_area);

	double part_2_1 = (uv.row(v_3)-uv.row(v_1)).dot(uv.row(v_2)-uv.row(v_1));
	double part_2_2 = m_cached_dot_prod[f_idx];
	double part_2 = -(part_2_1 * part_2_2)/ (orig_t_dbl_area);

	return part_1+part_2;
}*/

    return 0;
}