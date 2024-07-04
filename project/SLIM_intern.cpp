/**
 * Scalable Locally Injective Mappings
*/

#include "SLIM_intern.h"



TrianglesMapping::TrianglesMapping(const int acount, char** avariable) {
    Tut63(acount, avariable);
	// strcpy(energy, distortion);
}

Eigen::MatrixXd TrianglesMapping::getEigenMap() const {
	return EigenMap;
}

const char* TrianglesMapping::getOutput() const {
    return output_name;
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
            area[int(f)] = map.util.unsigned_area(f);
            vec2 A,B,C;
            map.util.project(f, A, B, C);

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
            Shape[int(f)] = S;
    }
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> TrianglesMapping::compute_gradients(double u1, double v1, double u2, double v2, double u3, double v3) {
		double A = 0.5 * std::fabs(u1 * v2 + u2 * v3 + u3 * v1 - u1 * v3 - u2 * v1 - u3 * v2);
		Eigen::Vector3d dudN, dvdN;
		dudN << (v2 - v3) / (2 * A), (v3 - v1) / (2 * A), (v1 - v2) / (2 * A);
		dvdN << (u3 - u2) / (2 * A), (u1 - u3) / (2 * A), (u2 - u1) / (2 * A);
		return std::make_pair(dudN, dvdN);
}

void TrianglesMapping::jacobian_rotation_area(Triangles& map, bool lineSearch) {
    num_vertices = map.nverts();
    num_triangles = map.nfacets();
    Dx = Eigen::SparseMatrix<double>(num_triangles, num_vertices);
    Dy = Eigen::SparseMatrix<double>(num_triangles, num_vertices);
    Af = Eigen::MatrixXd::Zero(num_triangles, num_triangles);
    if (!lineSearch) {xk_1 = Eigen::VectorXd::Zero(2 * num_triangles);}

    Jac.clear();
    Rot.clear();
    int ind = 0;
    std::vector<Eigen::Triplet<double>> Dx_triplets;
    std::vector<Eigen::Triplet<double>> Dy_triplets;

    for (auto f : map.iter_facets()) {
        Eigen::Matrix2d J_i;
        Eigen::Matrix<double, 3, 2> Z_i;
        Z_i << -1, -1,
               1, 0,
               0, 1;
        
        Z_i *= Shape[int(f)].inverse();

        //std::cout << "Z_i: " << std::endl << Z_i << std::endl;

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
                xk_1(v_ind + num_vertices) = f.vertex(j).pos()[2];
            }
        }

        J_i(0, 0) = f.vertex(1).pos()[0] - f.vertex(0).pos()[0];
        J_i(1, 0) = f.vertex(1).pos()[2] - f.vertex(0).pos()[2];
        J_i(0, 1) = f.vertex(2).pos()[0] - f.vertex(0).pos()[0];
        J_i(1, 1) = f.vertex(2).pos()[2] - f.vertex(0).pos()[2];

        J_i *= Shape[int(f)].inverse();
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

        Af(ind, ind) = std::sqrt(calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()));
        // Af(ind, ind) = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());

        ind++;
    }

    // Assemble the sparse matrices Dx and Dy
    Dx.setFromTriplets(Dx_triplets.begin(), Dx_triplets.end());
    Dy.setFromTriplets(Dy_triplets.begin(), Dy_triplets.end());
}

void TrianglesMapping::update_weights() {
	Wei.clear();
	if (strcmp(energy, "arap") == 0) {
		for (size_t i = 0; i < Rot.size(); ++i) {
			Wei.push_back(Eigen::Matrix2d::Identity());
		}
	}
	else if (strcmp(energy, "symm_dirichlet") == 0) {
		//
	}
	else if (strcmp(energy, "dmitry") == 0) {
		//
	}
}

void TrianglesMapping::least_squares() {
	Eigen::SparseMatrix<double> A(4 * num_triangles, 2 * num_vertices);
	Eigen::VectorXd b = Eigen::VectorXd::Zero(4 * num_triangles);

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

	// Fill in the A matrix
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < num_triangles; ++i) {
        double Af_i = Af(i, i); // Accessing the diagonal element of Af

        for (int k = 0; k < Dx.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Dx, k); it; ++it) {
                if (it.row() == i) { // Check if the current non-zero element belongs to the current triangle
                    int j = it.col();
                    triplets.push_back({i, j, Af_i * W11_diag(i) * it.value()});
                    triplets.push_back({i, num_vertices + j, Af_i * W12_diag(i) * it.value()});
                    triplets.push_back({num_triangles + i, j, Af_i * W21_diag(i) * it.value()});
                    triplets.push_back({num_triangles + i, num_vertices + j, Af_i * W22_diag(i) * it.value()});
                }
            }
        }

        for (int k = 0; k < Dy.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Dy, k); it; ++it) {
                if (it.row() == i) { // Check if the current non-zero element belongs to the current triangle
                    int j = it.col();
                    triplets.push_back({2 * num_triangles + i, j, Af_i * W11_diag(i) * it.value()});
                    triplets.push_back({2 * num_triangles + i, num_vertices + j, Af_i * W12_diag(i) * it.value()});
                    triplets.push_back({3 * num_triangles + i, j, Af_i * W21_diag(i) * it.value()});
                    triplets.push_back({3 * num_triangles + i, num_vertices + j, Af_i * W22_diag(i) * it.value()});
                }
            }
        }
    }

	A.setFromTriplets(triplets.begin(), triplets.end());

	// Fill in the b vector
	for (int i = 0; i < num_triangles; ++i) {
		double Af_i = Af(i, i);

		b(i) = Af_i * (W11_diag(i) * R11(i) + W12_diag(i) * R12(i));
		b(num_triangles + i) = Af_i * (W21_diag(i) * R11(i) + W22_diag(i) * R12(i));
		b(2 * num_triangles + i) = Af_i * (W11_diag(i) * R21(i) + W12_diag(i) * R22(i));
		b(3 * num_triangles + i) = Af_i * (W21_diag(i) * R21(i) + W22_diag(i) * R22(i));
	}

	// Use an iterative solver
	// Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> solver; // ConjugateGradient solver for symmetric positive definite matrices
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver; // BiCGSTAB solver for square matrices	
    std::cout << "1" << std::endl;
    double lambda = 0.0001;
    Eigen::SparseMatrix<double> AtA = A.transpose() * A + lambda * Eigen::MatrixXd::Identity(2 * num_vertices, 2 * num_vertices);
    solver.compute(AtA);
    std::cout << "2" << std::endl;
    // solver.compute(A.transpose() * A);

	if(solver.info() != Eigen::Success) {
		// Decomposition failed
		std::cerr << "Decomposition failed" << std::endl;
		return;
	}

    // xk = solver.solve(A.transpose() * b);
    Eigen::MatrixXd Atb = A.transpose() * b + lambda * xk_1;
    std::cout << "3" << std::endl;
	xk = solver.solve(Atb);
    std::cout << "4" << std::endl;

	if(solver.info() != Eigen::Success) {
		// Solving failed
		std::cerr << "Solving failed" << std::endl;
		return;
	}

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

	dk = xk - xk_1;
	std::cout << "Distance dk:" << std::endl << dk << std::endl;
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
            f.vertex(j).pos()[2] = xk(v_ind + num_vertices);
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

 double TrianglesMapping::minimum_step_singularities(Triangles& map, Eigen::VectorXd& x, Eigen::VectorXd& d) {
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
        #define U11 x(int(f.vertex(0)))
        #define U12 x(int(f.vertex(0)) + num_vertices)
        #define U21 x(int(f.vertex(1)))
        #define U22 x(int(f.vertex(1)) + num_vertices)
        #define U31 x(int(f.vertex(2)))
        #define U32 x(int(f.vertex(2)) + num_vertices)

        #define V11 d(int(f.vertex(0)))
        #define V12 d(int(f.vertex(0)) + num_vertices)
        #define V21 d(int(f.vertex(1)))
        #define V22 d(int(f.vertex(1)) + num_vertices)
        #define V31 d(int(f.vertex(2)))
        #define V32 d(int(f.vertex(2)) + num_vertices)
        
        
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

void TrianglesMapping::add_energies_jacobians(double& norm_arap_e, bool flips_linesearch) {
	// schaeffer_e = log_e = conf_e = amips = 0;
	norm_arap_e = 0;
	for (int i = 0; i < num_triangles; i++) {
		Eigen::JacobiSVD<Eigen::Matrix2d> svd(Jac[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix2d Ui = svd.matrixU();
		Eigen::Matrix2d Vi = svd.matrixV();
		Eigen::Vector2d singu = svd.singularValues();

		double s1 = singu(0); double s2 = singu(1);

		if (flips_linesearch) {
			// schaeffer_e += Af(i, i) * (pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2));
			// log_e += Af(i, i) * (pow(log(s1),2) + pow(log(s2),2));
			// double sigma_geo_avg = sqrt(s1*s2);
			// conf_e += Af(i, i) * (pow(log(s1/sigma_geo_avg),2) + pow(log(s2/sigma_geo_avg),2));
			// conf_e += Af(i, i) * ( (pow(s1,2)+pow(s2,2))/(2*s1*s2) );
			norm_arap_e += Af(i, i) * (pow(s1-1,2) + pow(s2-1,2));
			// amips +=  Af(i, i) * exp(exp_factor* (  0.5*( (s1/s2) +(s2/s1) ) + 0.25*( (s1*s2) + (1./(s1*s2)) )  ) );
			// exp_symmd += Af(i, i) * exp(exp_factor*(pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2)));
			// amips +=  Af(i, i) * exp(  0.5*( (s1/s2) +(s2/s1) ) + 0.25*( (s1*s2) + (1./(s1*s2)) )  ) ;
		} else {
			if (Ui.determinant() * Vi.determinant() > 0) {
			norm_arap_e += Af(i, i) * (pow(s1-1,2) + pow(s2-1,2));
			} else {
			// it is the distance form the flipped thing, this is slow, usefull only for debugging normal arap
			Vi.col(1) *= -1;
			norm_arap_e += Af(i, i) * (Jac[i]-Ui*Vi.transpose()).squaredNorm();
			}
		}
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
            pos(1) = facet_i.vertex(j).pos()[2];
            
            Eigen::Matrix2d grad_J = Eigen::Matrix2d::Zero();
            Eigen::Vector2d e1;
            Eigen::Vector2d e2;
            e1(0) = facet_i.vertex(1).pos()[0] - facet_i.vertex(0).pos()[0];
            e1(1) = facet_i.vertex(1).pos()[2] - facet_i.vertex(0).pos()[2];
            e2(0) = facet_i.vertex(2).pos()[0] - facet_i.vertex(0).pos()[0];
            e2(1) = facet_i.vertex(2).pos()[2] - facet_i.vertex(0).pos()[2];

            // Partial derivatives of e1 and e2 with respect to vertex positions
            Eigen::Matrix2d de1_dv0 = -Eigen::Matrix2d::Identity(); // Partial derivative of e1 w.r.t v0
            Eigen::Matrix2d de1_dv1 = Eigen::Matrix2d::Identity();  // Partial derivative of e1 w.r.t v1
            Eigen::Matrix2d de2_dv0 = -Eigen::Matrix2d::Identity(); // Partial derivative of e2 w.r.t v0
            Eigen::Matrix2d de2_dv2 = Eigen::Matrix2d::Identity();  // Partial derivative of e2 w.r.t v2

            grad_J += (de1_dv0 + de2_dv0); // Combine contributions from all vertices

            Eigen::Matrix2d dJ = grad_J * (Jac[i] - R_i);

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

double TrianglesMapping::lineSearch(Eigen::VectorXd& xk_search, Eigen::VectorXd& dk,
                      Triangles& map) {
    // Line search using Wolfe conditions
    double c1 = 1e-4; // 1e-5
    double c2 = 0.9; // 0.99
    std::cout << "lineSearch: " << std::endl;
    double alphaMax = minimum_step_singularities(map, xk_search, dk);
    // double alphaMax = determineAlphaMax(xk_search, dk, map);
    
    double alphaStep = 0.99 * alphaMax;
    alphaStep = std::min(1.0, 0.8 * alphaMax);
    double alphaTest = 0.99 * alphaMax;
    double alphaStep2 = std::min(1.0, 0.8 * alphaMax);

    Eigen::VectorXd pk = xk_search + alphaStep * dk;
    std::cout << "alphaStep: " << alphaStep << std::endl;

    updateUV(map, xk_search);
    jacobian_rotation_area(map, true);
    double ener, new_ener;
    add_energies_jacobians(ener, true);
    
    // Compute gradient of xk_search
    Eigen::VectorXd grad_xk = Eigen::VectorXd::Zero(xk_search.size());
    // computeGradient(xk_search, grad_xk, map);
	// computeAnalyticalGradient(xk_search, grad_xk, map);
    compute_energy_gradient(grad_xk, true, map);
    
    /*for (auto v : map.iter_vertices()) {
        v.pos()[0] = pk(int(v));
        v.pos()[2] = pk(int(v) + num_vertices);
    }*/
    Eigen::VectorXd grad_pk = Eigen::VectorXd::Zero(pk.size());

    // Wolfe conditions
    auto wolfe1 = [&]() {
        return new_ener <= ener + c1 * alphaStep * grad_xk.dot(dk);
    };

    auto wolfe2 = [&]() {
        return grad_pk.dot(dk) >= c2 * grad_xk.dot(dk);
    };

    // // Initial check
    // if (wolfe1() && wolfe2()) {
    //     return alpha;
    // }

    // Bisection line search with max_iter limit
    double alphaLow = 0.0;
    double alphaHigh = alphaMax;
    int max_iter = 40;
    int iter = 0;

    while (alphaHigh - alphaLow > 1e-8 && iter < max_iter) {
        if (!wolfe1()) {
            alphaHigh = alphaStep;
            alphaStep = (alphaLow + alphaHigh) / 2.0;
            
        } else if (!wolfe2()) {
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

        pk = xk_search + alphaStep * dk;

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


    while (alphaHigh - alphaLow > 1e-8 && iter < max_iter) {
        if (!(new_ener <= ener)) {
            alphaHigh = alphaStep2;
            alphaStep2 = (alphaLow + alphaHigh) / 2.0;
            
        } else {
            alphaLow = alphaStep2;
            alphaStep2 = (alphaLow + alphaHigh) / 2.0;
        }

        pk = xk_search + alphaStep2 * dk;

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

	std::cout << "The new solution is:\n" << xk << std::endl;
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
        area_b += calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
    }
    double radius = sqrt(area_b / (M_PI));

    int total_len = bound_sorted.size();
    int len_i = 0;
    for (auto v : bound_sorted) {
        double frac = len_i * (2. * M_PI) / total_len;
        map.points[v][0] = radius * cos(frac);
        map.points[v][2] = radius * sin(frac);
        len_i++;
    }
}

void TrianglesMapping::Tut63(const int acount, char** avariable) {
    const char* name = nullptr;
    int weights = -1;
    if (acount > 1) {
        for (int i = 1; i < acount; ++i) {
            if (strlen(avariable[i]) == 1) {
                weights = atoi(avariable[i]);
            } else if (strlen(avariable[i]) > 1) {
                name = avariable[i];
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

    if (weights == -1) {
        weights = 2;
    }

    std::filesystem::path filepath = name;
    std::string filepath_str_ext = filepath.extension().string();
    std::string filepath_str_stem = filepath.stem().string();
    const char* ext = filepath_str_ext.c_str();
    const char* stem = filepath_str_stem.c_str();

    char ext2[12] = ".geogram";
    char method[20] = "_barycentre";
    char weight1[20] = "_uniform";
    char weight2[20] = "_cotan";
    char attribute[20] = "_distortion";

    strcpy(output_name, stem);
    strcat(output_name, method);
    if (weights == 1) {
        strcat(output_name, weight1);
    } else if (weights == 2) {
        strcat(output_name, weight2);
    }
    strcat(output_name, attribute);
    strcat(output_name, ext2);

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
    double cuttingSurface = 0.;
    double margin = 1.0 / nverts * 100;
    double dcuttingSurface = (maxH - minH) * margin;

    DBOUT("The number of vertices is: " << mTut.nverts() << ", facets: " << mTut.nfacets() << ", corners: " << mTut.ncorners() << std::endl);

    mOri.connect();
    mTut.connect();

    reference_mesh(mOri);

    /*int fixed = 0;
    Eigen::VectorXd x_B_ = Eigen::VectorXd::Zero(nverts);
    Eigen::VectorXd x_I_ = Eigen::VectorXd::Zero(nverts);
    int insider = 0;*/
    std::unordered_map<int, std::vector<int>> neighbor;
    std::unordered_set<int> bound;

    for (auto he : mOri.iter_halfedges()) {
        if (!he.opposite().active()) {
            neighbor[he.from()].push_back(he.to());
            neighbor[he.to()].push_back(he.from());
            bound.insert(he.from());
            bound.insert(he.to());
        }
    }

    std::vector<int> bound_sorted;
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

    exit(EXIT_FAILURE);

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

    Eigen::MatrixXd lhsF = Eigen::MatrixXd::Zero(nverts, 3);

    bound_vertices_circle_normalized(mTut);

    int vert = 0;
    int bb = 0;
    double progress = 0;
    for (int i = 0; i < mTut.nverts(); i++) {
        Surface::Vertex vi = Surface::Vertex(mOri, i);
        if (bound.contains(vi)) {
            x_B(bb, 0) = mTut.points[i][0];
            x_B(bb, 1) = mTut.points[i][2];

            lhsF(bb, 0) = mTut.points[i][0];
            lhsF(bb, 1) = mTut.points[i][2];

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

   Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_II_A_BB);
    if (solver.info() != Eigen::Success) {
        // Decomposition failed
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }

    /*// Set higher precision by adjusting tolerance and max iterations
    solver.setTolerance(1e-10);  // Set a tighter tolerance
    solver.setMaxIterations(10000);  // Increase the maximum number of iterations*/

    Eigen::MatrixXd x_I_full = solver.solve(lhsF);
    if (solver.info() != Eigen::Success) {
        // Solving failed
        std::cerr << "Solving failed" << std::endl;
        return;
    }

    std::cout << x_I_full << std::endl;
    EigenMap = x_I_full;

    for (int plan : plane) {
        int re = x_I_(plan);
        // mTut.points[plan][0] = x_I(re, 0);
	    // mTut.points[plan][2] = x_I(re, 1);

        mTut.points[plan][0] = x_I_full(re + fixed, 0);
        mTut.points[plan][1] = cuttingSurface;
        mTut.points[plan][2] = x_I_full(re + fixed, 1);
    }
    for (int blad : blade) {
        mTut.points[blad][1] = cuttingSurface;
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
        double area = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
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

    FacetAttribute<double> fa(mTut);
    for (auto f : mTut.iter_facets()) {
        fa[f] = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fa2[f];
    }

    write_by_extension(output_name, mTut, { {}, {{"DistortionScale", fa.ptr}}, {{"Halfedge", he.ptr}} });

    #ifdef _WIN32
    // Open the generated mesh with Graphite
    int result = system((getGraphitePath() + " " + output_name).c_str());
    #endif
    #ifdef linux
    system((std::string("graphite ") + output_name).c_str());
    #endif
}

void TrianglesMapping::LocalGlobalParametrization(const char* map) {
    read_by_extension(map, mLocGlo);
    mLocGlo.connect();

    std::ofstream timeFile("iteration_times.txt"); // Open a file for writing times
    auto totalStart = std::chrono::high_resolution_clock::now(); // Start timing for total duration
    long long totalTime = 0; // Initialize total time accumulator

	for (int i = 0; i < max_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        jacobian_rotation_area(mLocGlo, false);
        std::cout << "jacobian_rotation_area(mLocGlo);" << std::endl;
        update_weights();
        std::cout << "update_weights();" << std::endl;
        least_squares();
        std::cout << "least_squares();" << std::endl;
        nextStep(mLocGlo);
        std::cout << "nextStep(mLocGlo);" << std::endl;

        std::filesystem::path filepath = map;
        std::string filepath_str_ext = filepath.extension().string();
        std::string filepath_str_stem = filepath.stem().string();
        const char* ext = filepath_str_ext.c_str();
        const char* stem = filepath_str_stem.c_str();

        char ext2[12] = ".geogram";
        char method[20] = "_local_global_";
        char attribute[20] = "_distortion_";
        char numStr[20];
        
        output_name[0] = '\0'; // Clear output_name
        const char* first_space = strchr(stem, '_'); // Find the first space in stem
        size_t first_word_length = first_space ? (size_t)(first_space - stem) : strlen(stem); // Calculate length of the first word
        strncpy(output_name, stem, first_word_length);
        output_name[first_word_length] = '\0'; // Ensure null-termination
        strcat(output_name, method);
        strcat(output_name, energy);
        strcat(output_name, attribute);
        sprintf(numStr, "%d", i);
        strcat(output_name, numStr);
        strcat(output_name, ext2);

        reference_mesh(mLocGlo);

        CornerAttribute<double> he(mLocGlo);
        for (auto f : mLocGlo.iter_halfedges()) {
            if (blade.contains(f.from()) || blade.contains(f.to())) {
                he[f] = 404;
            } else {
                he[f] = 0;
            }
        }

        FacetAttribute<double> fa(mLocGlo);
        for (auto f : mLocGlo.iter_facets()) {
            fa[f] = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos()) / fOriMap[int(f)];
        }


        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate duration in milliseconds
        totalTime += duration; // Accumulate total time

        if (timeFile.is_open()) {
            timeFile << "Iteration " << i << ": " << duration << " ms — "; // Write iteration number and duration to file
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count(); // Calculate total duration in milliseconds

        if (timeFile.is_open()) {
            timeFile << "Total time: " << totalDuration << " ms\n"; // Log total time
        }


        write_by_extension(output_name, mLocGlo, { {}, {{"DistortionScale", fa.ptr}}, {{"Halfedge", he.ptr}} });
        std::cout << "write_by_extension(output_name, mLocGlo);" << std::endl;
        if (i % 20 == 0) {
            #ifdef _WIN32
            // Open the generated mesh with Graphite
            int result = system((getGraphitePath() + " " + output_name).c_str());
            #endif
            #ifdef linux
            system((std::string("graphite ") + output_name).c_str());
            #endif
        }
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

	for (int progress = 0; progress <= 100; ++progress) {
        updateProgressBar(progress);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << std::endl;


	/*


	----------------------------------------3D to 2D----------------------------------------


Export .obj ou .geogram, coordonnées de texture pour graphite


//////////////////////////////////////////////////ENERGY//////////////////////////////////////////////////



void TrianglesMapping::computeAnalyticalGradient(const Eigen::VectorXd& x, Eigen::VectorXd& grad, Triangles& map) {
    // Ensure grad is the correct size
    grad.resize(x.size());
    grad.setZero();

    // Reset vertex positions according to x
    for (auto v : map.iter_vertices()) {
        v.pos()[0] = x(int(v));
        v.pos()[2] = x(int(v) + num_vertices);
    }

    // Compute areas and internal state needed for gradients
    jacobian_rotation_area(map, true); // Assuming this computes areas and caches necessary values

    // Calculate per-face area for original and current configurations
    Eigen::VectorXd dblArea_orig, dblArea_p;
    compute_double_area(map, dblArea_orig); // Custom function to compute double area for the original configuration
    compute_double_area(map, dblArea_p);    // Custom function to compute double area for the current configuration

    // Placeholder for energy per face computation
    Eigen::VectorXd m_cached_l_energy_per_face(map.num_faces());
    Eigen::VectorXd m_cached_r_energy_per_face(map.num_faces());

    // Placeholder for cotangent entries
    Eigen::MatrixXd m_cot_entries(map.num_faces(), 3);

    // Compute energy per face and cotangent entries
    for (int i = 0; i < map.num_faces(); ++i) {
        auto& face = map.get_face(i);

        // Compute edge vectors
        Eigen::Vector2d v0 = map.vertex(face.vertex_index(0)).pos();
        Eigen::Vector2d v1 = map.vertex(face.vertex_index(1)).pos();
        Eigen::Vector2d v2 = map.vertex(face.vertex_index(2)).pos();

        Eigen::Vector2d e0 = v1 - v0;
        Eigen::Vector2d e1 = v2 - v1;
        Eigen::Vector2d e2 = v0 - v2;

        // Compute cotangent entries
        m_cot_entries(i, 0) = compute_cotangent(e0, e2);
        m_cot_entries(i, 1) = compute_cotangent(e1, -e0);
        m_cot_entries(i, 2) = compute_cotangent(e2, -e1);

        // Compute energy per face (placeholder example)
        m_cached_l_energy_per_face(i) = e0.squaredNorm(); // Replace with actual energy computation
        m_cached_r_energy_per_face(i) = e1.squaredNorm(); // Replace with actual energy computation
    }

    // Adjust gradients based on area terms
    for (int i = 0; i < map.num_faces(); ++i) {
        double t_orig_area = dblArea_orig(i) / 2.0;
        double t_uv_area = dblArea_p(i) / 2.0;
        double left_grad_const = -pow(t_orig_area, 2) / pow(t_uv_area, 3);

        auto& face = map.get_face(i);
        for (int j = 0; j < 3; ++j) {
            int v1 = face.vertex_index(j);
            int v2 = face.vertex_index((j + 1) % 3);
            int v3 = face.vertex_index((j + 2) % 3);

            // Compute left gradient
            Eigen::Vector2d rotated_left_grad = left_grad_const * (map.vertex(v2).pos() - map.vertex(v3).pos());
            Eigen::Vector2d c_left_grad(rotated_left_grad.y(), -rotated_left_grad.x());

            // Compute right gradient
            Eigen::Vector2d c_right_grad = -m_cot_entries(i, v2) * map.vertex(v3).pos()
                                           - m_cot_entries(i, v3) * map.vertex(v2).pos()
                                           + (m_cot_entries(i, v2) + m_cot_entries(i, v3)) * map.vertex(v1).pos();

            // Product rule and accumulation
            grad(v1) -= (c_left_grad * m_cached_r_energy_per_face[i] + c_right_grad * m_cached_l_energy_per_face[i]);
        }
    }
}

double TrianglesMapping::compute_cotangent(const Eigen::Vector2d& e1, const Eigen::Vector2d& e2) {
    double dot_product = e1.dot(e2);
    double cross_product = e1.x() * e2.y() - e1.y() * e2.x();
    return dot_product / cross_product;
}

void TrianglesMapping::compute_double_area(Triangles& map, Eigen::VectorXd& dblArea) {
    // Custom function to compute the double area of each triangle in the mesh
    dblArea.resize(map.num_faces());
    for (int i = 0; i < map.num_faces(); ++i) {
        auto& face = map.get_face(i);
        Eigen::Vector2d v0 = map.vertex(face.vertex_index(0)).pos();
        Eigen::Vector2d v1 = map.vertex(face.vertex_index(1)).pos();
        Eigen::Vector2d v2 = map.vertex(face.vertex_index(2)).pos();
        dblArea(i) = std::abs((v1 - v0).x() * (v2 - v0).y() - (v2 - v0).x() * (v1 - v0).y());
    }
}*/

    return 0;
}