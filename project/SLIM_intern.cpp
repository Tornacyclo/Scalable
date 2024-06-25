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

std::pair<Eigen::Vector3d, Eigen::Vector3d> TrianglesMapping::compute_gradients(double u1, double v1, double u2, double v2, double u3, double v3) {
		double A = 0.5 * std::fabs(u1 * v2 + u2 * v3 + u3 * v1 - u1 * v3 - u2 * v1 - u3 * v2);
		Eigen::Vector3d dudN, dvdN;
		dudN << (v2 - v3) / (2 * A), (v3 - v1) / (2 * A), (v1 - v2) / (2 * A);
		dvdN << (u3 - u2) / (2 * A), (u1 - u3) / (2 * A), (u2 - u1) / (2 * A);
		return std::make_pair(dudN, dvdN);
}

void TrianglesMapping::jacobian_rotation_area(Triangles& map) {
	num_vertices = map.nverts();
	num_triangles = map.nfacets();
	Dx = Eigen::MatrixXd::Zero(num_triangles, num_vertices);
	Dy = Eigen::MatrixXd::Zero(num_triangles, num_vertices);
	Af = Eigen::MatrixXd::Zero(num_triangles, num_triangles);
	xk_1 = Eigen::VectorXd::Zero(2 * num_triangles);
	for (auto f : map.iter_facets()) {
		int ind = 0;
		Eigen::Matrix2d J_i;
		J_i(0, 0) = f.vertex(1).pos()[0] - f.vertex(0).pos()[0];
		J_i(0, 1) = f.vertex(2).pos()[0] - f.vertex(0).pos()[0];
		J_i(1, 0) = f.vertex(1).pos()[2] - f.vertex(0).pos()[2];
		J_i(1, 1) = f.vertex(2).pos()[2] - f.vertex(0).pos()[2];
		Jac.push_back(J_i);

		// Compute SVD of J_i
		Eigen::JacobiSVD<Eigen::Matrix2d> svd(J_i, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix2d U = svd.matrixU();
		Eigen::Matrix2d V = svd.matrixV();

		// Construct the closest rotation matrix R_i
		Eigen::Matrix2d R_i = U * V.transpose();

		// Store R_i in the vector
		Rot.push_back(R_i);

		
		double u1 = f.vertex(0).pos()[0], v1 = f.vertex(0).pos()[2];
		double u2 = f.vertex(1).pos()[0], v2 = f.vertex(0).pos()[2];
		double u3 = f.vertex(2).pos()[0], v3 = f.vertex(0).pos()[2];
		auto gradients = compute_gradients(u1, v1, u2, v2, u3, v3);
		Eigen::Vector3d dudN = gradients.first;
		Eigen::Vector3d dvdN = gradients.second;
		for (int j = 0; j < 3; ++j) {
			Dx(ind, int(f.vertex(j))) = dudN(j);
			Dy(ind, int(f.vertex(j))) = dvdN(j);

			xk_1(int(f.vertex(j))) = f.vertex(j).pos()[0];
			xk_1(int(f.vertex(j)) + num_vertices) = f.vertex(j).pos()[2];
		}

		Af(ind, ind) = std::sqrt(calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos())); // Compute the square root of the area of the triangle

		ind++;
	}
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
	Eigen::MatrixXd W11 = Eigen::MatrixXd::Zero(num_triangles, num_triangles);
	Eigen::MatrixXd W12 = Eigen::MatrixXd::Zero(num_triangles, num_triangles);
	Eigen::MatrixXd W21 = Eigen::MatrixXd::Zero(num_triangles, num_triangles);
	Eigen::MatrixXd W22 = Eigen::MatrixXd::Zero(num_triangles, num_triangles);

	Eigen::VectorXd R11 = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd R12 = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd R21 = Eigen::VectorXd::Zero(num_triangles);
	Eigen::VectorXd R22 = Eigen::VectorXd::Zero(num_triangles);

	for (int i = 0; i < num_triangles; ++i) {
		W11(i, i) = Wei[i](0, 0);
		W12(i, i) = Wei[i](0, 1);
		W21(i, i) = Wei[i](1, 0);
		W22(i, i) = Wei[i](1, 1);

		R11(i, 0) = Rot[i](0, 0);
		R12(i, 0) = Rot[i](0, 1);
		R21(i, 0) = Rot[i](1, 0);
		R22(i, 0) = Rot[i](1, 1);
	}

	// Form A and b matrices
	Eigen::MatrixXd A(4 * num_triangles, 2 * num_vertices);
	A << Af * W11 * Dx, Af * W12 * Dx,
		Af * W21 * Dx, Af * W22 * Dx,
		Af * W11 * Dy, Af * W12 * Dy,
		Af * W21 * Dy, Af * W22 * Dy;
	
	Eigen::VectorXd b(4 * num_triangles);
	b << Af * W11 * R11 + Af * W12 * R12,
		Af * W21 * R11 + Af * W22 * R12,
		Af * W11 * R21 + Af * W12 * R22,
		Af * W21 * R21, Af * W22 * R22;

	// Solve using least squares
	// x = A.colPivHouseholderQr().solve(b);
	// Eigen::MatrixXd At = A.transpose();
	// Eigen::MatrixXd AtA = At * A;
	// Eigen::MatrixXd AtB = At * b;
	// Eigen::MatrixXd X = AtA.colPivHouseholderQr().solve(AtB);
	// double lambda = 0.0001;
	// // Solve for pk (argmin problem)
	// Eigen::VectorXd pk = (A.transpose() * A + lambda * Eigen::MatrixXd::Identity(2 * num_vertices, 2 * num_vertices)).ldlt().solve(A.transpose() * b + lambda * xk_1);
	xk = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

	std::cout << "Solution x:" << std::endl << xk << std::endl << xk.rows() << std::endl << xk.cols() << std::endl;
}

double TrianglesMapping::lineSearch(const Eigen::VectorXd& xk, const Eigen::VectorXd& dk,
                      std::function<double(const Eigen::VectorXd&)> objFunc,
                      std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradFunc) {
	// Line search using Wolfe conditions
	double alpha = 1.0;
	double c1 = 1e-4;
	double c2 = 0.9;
	double alphaMax = 1.0;

	Eigen::VectorXd pk = xk + alpha * dk;

	// Wolfe conditions
	auto wolfe1 = [&]() {
		return objFunc(pk) <= objFunc(xk) + c1 * alpha * gradFunc(xk).dot(dk);
	};

	auto wolfe2 = [&]() {
		return gradFunc(pk).dot(dk) >= c2 * gradFunc(xk).dot(dk);
	};

	// Initial check
	if (wolfe1() && wolfe2()) {
		return alpha;
	}

	// Bisection line search
	double alphaLow = 0.0;
	double alphaHigh = alphaMax;

	while (alphaHigh - alphaLow > 1e-8) {
		alpha = (alphaLow + alphaHigh) / 2.0;
		pk = xk + alpha * dk;

		if (!wolfe1()) {
			alphaHigh = alpha;
		} else if (!wolfe2()) {
			alphaLow = alpha;
		} else {
			break;
		}
	}

	return alpha;
}

void TrianglesMapping::nextStep() {
	// Define the objective function
	auto objFunc = [](const Eigen::VectorXd& x) -> double {
		return x.squaredNorm();  // Example: simple quadratic function
	};

	// Define the gradient function
	auto gradFunc = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
		return 2 * x;  // Example: gradient of simple quadratic function
	};

	// Example starting point and direction
	Eigen::VectorXd xk = Eigen::VectorXd::Random(3);  // Random starting point
	Eigen::VectorXd dk = Eigen::VectorXd::Ones(3);    // Example direction

	// Perform line search to find step size alpha
	double alpha = lineSearch(xk, dk, objFunc, gradFunc);

	// Update the solution xk
	xk = xk + alpha * dk;

	// Output the result
	std::cout << "The new solution is:\n" << xk << std::endl;
	std::cout << "Step size alpha: " << alpha << std::endl;


	for (auto v : mLocGlo.iter_vertices()) {
		v.pos()[0] = xk(int(v));
		v.pos()[2] = xk(int(v) + num_vertices);
	}
}

void TrianglesMapping::Tut63(const int acount, char** avariable) {
	const char* name = nullptr; int weights = -1;
	if (acount > 1) {
		for (int i = 1; i < acount; ++i) {
			if (strlen(avariable[i]) == 1) {
				weights = atoi(avariable[i]);
			}
			else if (strlen(avariable[i]) > 1) {
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
		weights = 1;
	}
	
	std::filesystem::path filepath = name;
	std::string filepath_str_ext = filepath.extension().string();
	std::string filepath_str_stem = filepath.stem().string();
	const char* ext = filepath_str_ext.c_str();
	stem = filepath_str_stem.c_str();
	
	char ext2[12] = ".geogram";
	char method[20] = "_barycentre";
	char weight1[20] = "_uniform";
	char weight2[20] = "_cotan";
	char attribute[20] = "_distortion";
	
	
	strcpy(output_name, stem);
	strcat(output_name, method);
	if (weights == 1) {
	strcat(output_name, weight1);
	}
	else if (weights == 2) {
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
	}
	else if (mTut.points[i][1] > maxH) {
		maxH = mTut.points[i][1];
	}
	}
	double cuttingSurface = 0.;
	double margin = 1.0 / nverts * 100;
	double dcuttingSurface = (maxH - minH)*margin;

	DBOUT("The number of vertices is: " << mTut.nverts() << ", facets: " << mTut.nfacets() << ", corners: " << mTut.ncorners() << std::endl);

	mOri.connect();
	mTut.connect();

	int fixed = 0;
	std::set<int> blade;
	Eigen::MatrixXd x_B_ = Eigen::MatrixXd::Zero(nverts, 1);
	for (int i = 0; i < mTut.nverts(); i++) {
	Surface::Vertex vi = Surface::Vertex(mOri, i);
	if (vi.pos()[1] <= cuttingSurface+dcuttingSurface && vi.pos()[1] >= cuttingSurface-dcuttingSurface) {
	    
	    Surface::Vertex vi = Surface::Vertex(mOri, i);
	    blade.insert(vi);
	    x_B_(i, 0) = fixed;
	    fixed++;
	}
	}

	Eigen::MatrixXd x_I_ = Eigen::MatrixXd::Zero(nverts, 1);
	int insider = 0;
	std::set<int> plane;
	for (int i = 0; i < mTut.nverts(); i++) {
	Surface::Vertex vi = Surface::Vertex(mOri, i);
	if (vi.pos()[1] <= cuttingSurface-dcuttingSurface || vi.pos()[1] >= cuttingSurface+dcuttingSurface) {
	    
	    plane.insert(vi);
	    x_I_(i, 0) = insider;
	    insider++;
	}
	}

	Eigen::MatrixXd A_II = Eigen::MatrixXd::Zero(nverts-fixed, nverts-fixed);
	Eigen::MatrixXd A_IB = Eigen::MatrixXd::Zero(nverts-fixed, fixed);
	Eigen::MatrixXd b_I = Eigen::MatrixXd::Zero(nverts - fixed, 2);
	Eigen::MatrixXd x_B = Eigen::MatrixXd::Zero(fixed, 2);



	Eigen::MatrixXd A_II_A_BB = Eigen::MatrixXd::Zero(nverts, nverts);
	for (int i = 0; i < fixed; ++i) {
	A_II_A_BB(i, i) = 1;
	}

	Eigen::MatrixXd lhsF = Eigen::MatrixXd::Zero(nverts, 3);



	int vert = 0;
	int bb = 0;
	double progress = 0;
	for (int i = 0; i < mTut.nverts(); i++) {
	Surface::Vertex vi = Surface::Vertex(mOri, i);
	if (vi.pos()[1] <= cuttingSurface+dcuttingSurface && vi.pos()[1] >= cuttingSurface-dcuttingSurface) {
	    x_B(bb, 0) = mOri.points[i][0];
	    x_B(bb, 1) = mOri.points[i][2];
	    
	    
	    lhsF(bb, 0) = mOri.points[i][0];
	    lhsF(bb, 1) = mOri.points[i][2];
	    
	    
	    bb++;
	}
	else {
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

		    int ree = x_I_(i, 0);
		    A_II(ree, ree) = -count;
		    
		    
		    A_II_A_BB(ree + fixed, ree + fixed) = -count;
		    
		    for (auto const& [key, val] : cotan)
			{
				if (blade.contains(key)) {
				    int re_ne2 = x_B_(key, 0);
				    A_IB(ree, re_ne2) = val;
				    
				    
				    A_II_A_BB(ree+fixed, re_ne2) = val;
				}
				else {
				    int re_ne = x_I_(key, 0);
				    A_II(ree, re_ne) = val;
				    
				    
				    A_II_A_BB(ree + fixed, re_ne + fixed) = val;
				}
			}
		    
	    }
	    
		else if (weights == 2) {
		    std::map<int, double> cotan;
		    if (depart.opposite().active())
			variable = variable.opposite().next();
		    
		    
		    double cotan_alpha = calculateCotan(depart.next().from().pos(), depart.next().to().pos(), depart.prev().to().pos(), depart.prev().from().pos());
		    double cotan_beta = calculateCotan(depart.opposite().prev().to().pos(), depart.opposite().prev().from().pos(), depart.opposite().next().from().pos(), depart.opposite().next().to().pos());
		    double cotan_gamma = calculateCotan(depart.next().to().pos(), depart.next().from().pos(), depart.from().pos(), depart.to().pos());
		    
		    double w_ij = cotan_alpha + cotan_beta;
		    double voronoi = 0.125 * (cotan_gamma*(depart.from().pos() - depart.to().pos()).norm2() + cotan_alpha*(depart.prev().to().pos() - depart.prev().from().pos()).norm2());
		    
		    neighbors.push_back(depart.to());
		    cotan.insert(std::make_pair(neighbors.back(), w_ij));
		    count += w_ij;
		    
		    while (depart != variable && variable.active()) {
			cotan_alpha = calculateCotan(variable.next().from().pos(), variable.next().to().pos(), variable.prev().to().pos(), variable.prev().from().pos());
			cotan_beta = calculateCotan(variable.opposite().prev().to().pos(), variable.opposite().prev().from().pos(), variable.opposite().next().from().pos(), variable.opposite().next().to().pos());
			cotan_gamma = calculateCotan(variable.next().to().pos(), variable.next().from().pos(), variable.from().pos(), variable.to().pos());
			
			w_ij = cotan_alpha + cotan_beta;
			voronoi += 0.125 * (cotan_gamma*(variable.from().pos() - variable.to().pos()).norm2() + cotan_alpha*(variable.prev().to().pos() - variable.prev().from().pos()).norm2());
			
			neighbors.push_back(variable.to());
			cotan.insert(std::make_pair(neighbors.back(), w_ij));
			count += w_ij;
			if (!variable.opposite().active())
			    break;
			variable = variable.opposite().next();
		    }

		    int ree = x_I_(i, 0);
		    A_II(ree, ree) = -count / (2*voronoi);
		    
		    
		    A_II_A_BB(ree + fixed, ree + fixed) = -count / (2*voronoi);
		    
		    for (auto const& [key, val] : cotan)
			{
				if (blade.contains(key)) {
				    int re_ne2 = x_B_(key, 0);
				    A_IB(ree, re_ne2) = val / (2*voronoi);
				    
				    
				    A_II_A_BB(ree+fixed, re_ne2) = val / (2*voronoi);
				}
				else {
				    int re_ne = x_I_(key, 0);
				    A_II(ree, re_ne) = val / (2*voronoi);
				    
				    
				    A_II_A_BB(ree + fixed, re_ne + fixed) = val / (2*voronoi);
				}
			}
		}
	    
	}
	vert++;
	progress = std::round(static_cast<float>(vert) / nverts * 100000.0f) / 100000.0f * 100;
	DBOUT("Vertex " << vert << "/" << nverts << " (" << progress << " %) --- dim1: " << mOri.points[i][0] << ", dim2: " << mOri.points[i][1] << ", dim3: " << mOri.points[i][2] << std::endl);
	}

	/*Eigen::MatrixXd lhs = b_I - A_IB * x_B;
	Eigen::MatrixXd x_I = A_II.colPivHouseholderQr().solve(lhs);*/
	
	Eigen::MatrixXd x_I = A_II_A_BB.colPivHouseholderQr().solve(lhsF);

	EigenMap = x_I;

	for (int plan : plane) {
	int re = (x_I_(plan, 0));
	/*mTut.points[plan][0] = x_I(re, 0);
	mTut.points[plan][2] = x_I(re, 1);*/
	mTut.points[plan][0] = x_I(re + fixed, 0);
	mTut.points[plan][1] = cuttingSurface;
	mTut.points[plan][2] = x_I(re + fixed, 1);
	}
	for (int blad : blade) {
	mTut.points[blad][1] = cuttingSurface;
	}
	/*for (int plan : plane) {
	int re = (x_I_(plan, 0));
	mTut.points[plan][0] = x_I(re + fixed, 0);
	mTut.points[plan][1] = x_I(re + fixed, 2);
	mTut.points[plan][2] = x_I(re + fixed, 1);
	}
	for (int blad : blade) {
	int re = (x_B_(blad, 0));
	mTut.points[blad][1] = x_I(re, 2);
	}*/
	FacetAttribute<double> fa2(mOri);
	for (auto f : mOri.iter_facets()) {
	fa2[f] = calculateTriangleArea(f.vertex(0).pos(), f.vertex(1).pos(), f.vertex(2).pos());
	}

	CornerAttribute<double> he(mTut);
	for (auto f : mTut.iter_halfedges()) {
	if (blade.contains(f.from()) || blade.contains(f.to()))
		he[f] = 404;
	else {
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
	for (int i = 0; i < max_iterations; ++i) {
    jacobian_rotation_area(mLocGlo);
    update_weights();
    least_squares();
    nextStep();

	char ext2[12] = ".geogram";
	char method[20] = "_local_global";
	char attribute[20] = "_distortion_";
	char numStr[20];
	
	strcpy(output_name, stem);
	strcat(output_name, method);
	strcat(output_name, energy);
	strcat(output_name, attribute);
	sprintf(numStr, "%d", i);
	strcat(output_name, numStr);
	strcat(output_name, ext2);

	write_by_extension(output_name, mLocGlo);
	#ifdef _WIN32
	// Open the generated mesh with Graphite
	int result = system((getGraphitePath() + " " + output_name).c_str());
	#endif
	#ifdef linux
	system((std::string("graphite ") + output_name).c_str());
	#endif
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


	/*using namespace Eigen;

    // Define the dimensions of the problem
    const int n = 4;  // Number of elements, adjust as needed

    // Define the matrices Dx and Dy (FE gradient matrices) - These should be defined based on your specific mesh
    MatrixXd Dx(n, n);
    MatrixXd Dy(n, n);

    // Example initialization (replace with actual Dx and Dy)
    Dx.setIdentity();
    Dy.setIdentity();

    // Define the diagonal matrices Wij
    VectorXd W11(n);
    VectorXd W12(n);
    VectorXd W21(n);
    VectorXd W22(n);

    // Example initialization (replace with actual values)
    W11.setOnes();
    W12.setOnes();
    W21.setOnes();
    W22.setOnes();

    // Create the diagonal matrices
    MatrixXd W11_diag = W11.asDiagonal();
    MatrixXd W12_diag = W12.asDiagonal();
    MatrixXd W21_diag = W21.asDiagonal();
    MatrixXd W22_diag = W22.asDiagonal();

    // Define matrix A
    MatrixXd A(4 * n, 2 * n);
    A << W11_diag * Dx, W12_diag * Dx,
         W21_diag * Dy, W22_diag * Dy,
         W11_diag * Dx, W12_diag * Dy,
         W21_diag * Dy, W22_diag * Dy;

    // Define vector b
    VectorXd R11(n);
    VectorXd R12(n);
    VectorXd R21(n);
    VectorXd R22(n);

    // Example initialization (replace with actual values)
    R11 << 1, 2, 3, 4;
    R12 << 1, 2, 3, 4;
    R21 << 1, 2, 3, 4;
    R22 << 1, 2, 3, 4;

    VectorXd b(4 * n);
    b << R11, R21, R12, R22;

    // Solve the least squares problem
    VectorXd x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);

    // Output the result
    std::cout << "The solution is:\n" << x << std::endl;



	
	using namespace Eigen;
using namespace std;

// Function to retrieve the (x, y) coordinates from the solution vector
vector<pair<double, double>> getCoordinates(const VectorXd& x) {
    int n = x.size() / 2;
    vector<pair<double, double>> coordinates(n);

    for (int i = 0; i < n; ++i) {
        double xCoord = x(i);
        double yCoord = x(i + n);
        coordinates[i] = make_pair(xCoord, yCoord);
    }

    return coordinates;
}

int main() {
    // Example dimension (number of vertices)
    const int n = 4;

    // Example solution vector x of dimension 2n
    VectorXd x(2 * n);
    x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;  // Replace with your actual solution vector

    // Retrieve the (x, y) coordinates
    vector<pair<double, double>> coordinates = getCoordinates(x);

    // Output the coordinates
    cout << "Coordinates of the vertices:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "Vertex " << i + 1 << ": (" << coordinates[i].first << ", " << coordinates[i].second << ")" << endl;
    }

    return 0;


	----------------------------------------3D to 2D----------------------------------------

	#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

Matrix3d computeJacobian(const Vector3d &A, const Vector3d &B, const Vector3d &C, const Vector2d &A_prime, const Vector2d &B_prime, const Vector2d &C_prime) {
    Matrix3d J;

    // Compute the edges in 3D
    Vector3d AB = B - A;
    Vector3d AC = C - A;

    // Compute the edges in 2D
    Vector2d AB_prime = B_prime - A_prime;
    Vector2d AC_prime = C_prime - A_prime;

    // Set up the Jacobian matrix
    J(0, 0) = AB_prime.x() / AB.x();
    J(0, 1) = AB_prime.x() / AB.y();
    J(0, 2) = AB_prime.x() / AB.z();

    J(1, 0) = AB_prime.y() / AB.x();
    J(1, 1) = AB_prime.y() / AB.y();
    J(1, 2) = AB_prime.y() / AB.z();

    J(2, 0) = AC_prime.x() / AC.x();
    J(2, 1) = AC_prime.x() / AC.y();
    J(2, 2) = AC_prime.x() / AC.z();

    return J;
}

int main() {
    // Define the vertices of the 3D triangle
    Vector3d A(0.0, 0.0, 0.0);
    Vector3d B(1.0, 0.0, 0.0);
    Vector3d C(0.0, 1.0, 0.0);

    // Define the vertices of the 2D triangle (deformation)
    Vector2d A_prime(0.0, 0.0);
    Vector2d B_prime(1.0, 0.0);
    Vector2d C_prime(0.0, 1.0);

    // Compute the Jacobian
    Matrix3d J = computeJacobian(A, B, C, A_prime, B_prime, C_prime);

    std::cout << "Jacobian Matrix:\n" << J << std::endl;

    // Compute the inverse of the Jacobian
    Matrix3d J_inv = J.inverse();

    std::cout << "Inverse Jacobian Matrix:\n" << J_inv << std::endl;

    return 0;
}
#include <vector>
#include <Eigen/Dense>

// Assuming Vector3d and Vector2d from Eigen are used

struct Triangle3D {
    Vector3d A, B, C;
};

struct Triangle2D {
    Vector2d A, B, C;
};

void computeMeshJacobian(const std::vector<Triangle3D> &triangles3D, const std::vector<Triangle2D> &triangles2D, std::vector<Matrix3d> &jacobians, std::vector<Matrix3d> &inverses) {
    size_t n = triangles3D.size();
    jacobians.resize(n);
    inverses.resize(n);

    for (size_t i = 0; i < n; ++i) {
        jacobians[i] = computeJacobian(triangles3D[i].A, triangles3D[i].B, triangles3D[i].C, triangles2D[i].A, triangles2D[i].B, triangles2D[i].C);
        inverses[i] = jacobians[i].inverse();
    }
}

int main() {
    // Define a mesh of 3D and 2D triangles
    std::vector<Triangle3D> triangles3D = { ... }; // Fill with 3D triangles
    std::vector<Triangle2D> triangles2D = { ... }; // Fill with corresponding 2D triangles

    std::vector<Matrix3d> jacobians;
    std::vector<Matrix3d> inverses;

    // Compute Jacobians and their inverses for the mesh
    computeMeshJacobian(triangles3D, triangles2D, jacobians, inverses);

    // Output results
    for (size_t i = 0; i < jacobians.size(); ++i) {
        std::cout << "Jacobian Matrix for triangle " << i << ":\n" << jacobians[i] << std::endl;
        std::cout << "Inverse Jacobian Matrix for triangle " << i << ":\n" << inverses[i] << std::endl;
    }

    return 0;
}

}*//////////////////////


	/*void map_vertices_to_circle_area_normalized(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const Eigen::VectorXi& bnd,
	Eigen::MatrixXd& UV) {
	
	Eigen::VectorXd dblArea_orig; // TODO: remove me later, waste of computations
	igl::doublearea(V,F, dblArea_orig);
	double area = dblArea_orig.sum()/2;
	double radius = sqrt(area / (M_PI));
	cout << "map_vertices_to_circle_area_normalized, area = " << area << " radius = " << radius << endl;

	// Get sorted list of boundary vertices
	std::vector<int> interior,map_ij;
	map_ij.resize(V.rows());
	interior.reserve(V.rows()-bnd.size());

	std::vector<bool> isOnBnd(V.rows(),false);
	for (int i = 0; i < bnd.size(); i++)
	{
		isOnBnd[bnd[i]] = true;
		map_ij[bnd[i]] = i;
	}

	for (int i = 0; i < (int)isOnBnd.size(); i++)
	{
		if (!isOnBnd[i])
		{
		map_ij[i] = interior.size();
		interior.push_back(i);
		}
	}

	// Map boundary to unit circle
	std::vector<double> len(bnd.size());
	len[0] = 0.;

	for (int i = 1; i < bnd.size(); i++)
	{
		len[i] = len[i-1] + (V.row(bnd[i-1]) - V.row(bnd[i])).norm();
	}
	double total_len = len[len.size()-1] + (V.row(bnd[0]) - V.row(bnd[bnd.size()-1])).norm();

	UV.resize(bnd.size(),2);
	for (int i = 0; i < bnd.size(); i++)
	{
		double frac = len[i] * (2. * M_PI) / total_len;
		UV.row(map_ij[bnd[i]]) << radius*cos(frac), radius*sin(frac);
	}

	}

	void get_flips(const Eigen::MatrixXd& V,
				const Eigen::MatrixXi& F,
				const Eigen::MatrixXd& uv,
				std::vector<int>& flip_idx) {
	flip_idx.resize(0);
	for (int i = 0; i < F.rows(); i++) {

		Eigen::Vector2d v1_n = uv.row(F(i,0)); Eigen::Vector2d v2_n = uv.row(F(i,1)); Eigen::Vector2d v3_n = uv.row(F(i,2));

		Eigen::MatrixXd T2_Homo(3,3);
		T2_Homo.col(0) << v1_n(0),v1_n(1),1;
		T2_Homo.col(1) << v2_n(0),v2_n(1),1;
		T2_Homo.col(2) << v3_n(0),v3_n(1),1;
		double det = T2_Homo.determinant();
		assert (det == det);
		if (det < 0) {
		//cout << "flip at face #" << i << " det = " << T2_Homo.determinant() << endl;
		flip_idx.push_back(i);
		}
	}
	}
	int count_flips(const Eigen::MatrixXd& V,
				const Eigen::MatrixXi& F,
				const Eigen::MatrixXd& uv) {

	std::vector<int> flip_idx;
	get_flips(V,F,uv,flip_idx);

	
	return flip_idx.size();
	}*/
    
    
    
    /*Eigen::MatrixXd lhsF2 = Eigen::MatrixXd::Zero(nverts, 2);
    
    
    
    
    
    
    for (auto f : m.iter_facets()) {
	Surface::Halfedge depart = Surface::Vertex(mTut, i).halfedge();
	Surface::Halfedge variable = depart;
	int count = 0;
	
	std::vector<int> neighbors;
	std::map<int, double> cotan;
	if (depart.opposite().active())
	variable = variable.opposite().next();
	vec3 v = depart.next().from().pos() - depart.next().to().pos();
	vec3 w = depart.prev().to().pos() - depart.prev().from().pos();
	vec3 x = depart.opposite().prev().to().pos() - depart.opposite().prev().from().pos();
	vec3 y = depart.opposite().next().from().pos() - depart.opposite().next().to().pos();

	// double w_ij = 0.5 * (v*w / cross(v, w).norm() + x*y / cross(x, y).norm()) * (depart.from().pos() - depart.to().pos()).norm();

	// double w_ij = 1 / (v*w / cross(v, w).norm()) + 1 / (x*y / cross(x, y).norm());

	double w_ij = 1 / (depart.from().pos() - depart.to().pos()).norm();

	double voronoi = 0.125 * (v*w / cross(v, w).norm() + x*y / cross(x, y).norm()) * (depart.from().pos() - depart.to().pos()).norm2();
	neighbors.push_back(depart.to());
	cotan.insert(std::make_pair(neighbors.back(), w_ij));
	count += w_ij;
	while (depart != variable && variable.active()) {
	v = variable.next().from().pos() - variable.next().to().pos();
	w = variable.prev().to().pos() - variable.prev().from().pos();
	x = variable.opposite().prev().to().pos() - variable.opposite().prev().from().pos();
	y = variable.opposite().next().from().pos() - variable.opposite().next().to().pos();
	// w_ij = 0.5 * (v*w / cross(v, w).norm() + x*y / cross(x, y).norm()) * (variable.from().pos() - variable.to().pos()).norm();

	// w_ij = 1 / (v*w / cross(v, w).norm()) + 1 / (x*y / cross(x, y).norm());

	w_ij = 1 / (variable.from().pos() - variable.to().pos()).norm();

	voronoi += 0.125 * (v*w / cross(v, w).norm() + x*y / cross(x, y).norm()) * (variable.from().pos() - variable.to().pos()).norm2();

	neighbors.push_back(variable.to());
	cotan.insert(std::make_pair(neighbors.back(), w_ij));
	count += w_ij;
	if (!variable.opposite().active())
	    break;
	variable = variable.opposite().next();
	}
	
	
	
	if (plane.contains(i)){
		int ree = x_I_(i, 0);


		A_II_A_BB(ree + fixed, ree + fixed) = -count; // * 1/voronoi;
		for (int neighbor : neighbors) {
			if (blade.contains(neighbor)) {
			    int re_ne2 = x_B_(neighbor, 0);
			    A_IB(ree, re_ne2) = 1;
			    
			    
			    A_II_A_BB(ree+fixed, re_ne2) = 1;
			}
			else {
			    int re_ne = x_I_(neighbor, 0);
			    A_II(ree, re_ne) = 1;
			    
			    
			    A_II_A_BB(ree + fixed, re_ne + fixed) = 1;
			}

		}

		for (auto const& [key, val] : cotan)
		{
			if (blade.contains(key)) {
			    int re_ne2 = x_B_(key, 0);
			    A_IB(ree, re_ne2) = val; // * 1/voronoi;
			    
			    
			    A_II_A_BB(ree+fixed, re_ne2) = val; // * 1/voronoi;
			}
			else {
			    int re_ne = x_I_(key, 0);
			    A_II(ree, re_ne) = val; // * 1/voronoi;
			    
			    
			    A_II_A_BB(ree + fixed, re_ne + fixed) = val; // * 1/voronoi;
			}
		}
	}
	
	else {
		int ree = x_B_(i, 0);


		A_II_A_BB(ree, ree) = -count; // * 1/voronoi;
		for (int neighbor : neighbors) {
			if (blade.contains(neighbor)) {
			    int re_ne2 = x_B_(neighbor, 0);
			    
			    
			    A_II_A_BB(ree, re_ne2) = 1;
			}
			else {
			    int re_ne = x_I_(neighbor, 0);
			    
			    
			    A_II_A_BB(ree + fixed, re_ne + fixed) = 1;
			}

		}

		for (auto const& [key, val] : cotan)
		{
			if (blade.contains(key)) {
			    int re_ne2 = x_B_(key, 0); ///////////////////
			    
			    
			    A_II_A_BB(ree, re_ne2) = val; // * 1/voronoi;
			}
			else {
			    int re_ne = x_I_(key, 0);
			    A_II(ree, re_ne) = val; // * 1/voronoi;
			    
			    
			    A_II_A_BB(ree + fixed, re_ne + fixed) = val; // * 1/voronoi;
			}
		}
	}

            
        
        vert++;
        DBOUT("We are at vert " << vert << " " << mOri.points[i][0] << " " << mOri.points[i][2] << std::endl);
    }
    
    
    
    x_I = A_II_A_BB.colPivHouseholderQr().solve(lhsF);
    for (int plan : plane) {
        int re = (x_I_(plan, 0));
        mTut.points[plan][0] = x_I(re + fixed, 0);
        mTut.points[plan][2] = x_I(re + fixed, 1);
    }
    
    for (int blad : blade) {
    	int re = (x_B_(blad, 0)); ///////////////////
        mTut.points[blad][0] = x_I(re, 0);
        mTut.points[blad][2] = x_I(re, 1);
    }*/
    
    
    
    



    return 0;
}
