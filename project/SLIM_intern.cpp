/**
 * Scalable Locally Injective Mappings
*/

#include "SLIM_intern.h"



TrianglesMapping::TrianglesMapping(const int acount, char** avariable, const char* distortion) {
    Tut63(acount, avariable);
	strcpy(energy, distortion);
}

Eigen::MatrixXf TrianglesMapping::getEigenMap() const {
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

void TrianglesMapping::jacobians(Triangles& map) {
	for (auto f : map.iter_facets()) {
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

		// Step 4: Store R_i in the vector
		Rot.push_back(R_i);
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

void least_squares(const Eigen::MatrixXf& A, const Eigen::MatrixXf& b) {
	Eigen::MatrixXf At = A.transpose();
	Eigen::MatrixXf AtA = At * A;
	Eigen::MatrixXf AtB = At * B;
	Eigen::MatrixXf X = AtA.colPivHouseholderQr().solve(AtB);
	
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
	Eigen::MatrixXf x_B_ = Eigen::MatrixXf::Zero(nverts, 1);
	for (int i = 0; i < mTut.nverts(); i++) {
	Surface::Vertex vi = Surface::Vertex(mOri, i);
	if (vi.pos()[1] <= cuttingSurface+dcuttingSurface && vi.pos()[1] >= cuttingSurface-dcuttingSurface) {
	    
	    Surface::Vertex vi = Surface::Vertex(mOri, i);
	    blade.insert(vi);
	    x_B_(i, 0) = fixed;
	    fixed++;
	}
	}

	Eigen::MatrixXf x_I_ = Eigen::MatrixXf::Zero(nverts, 1);
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

	Eigen::MatrixXf A_II = Eigen::MatrixXf::Zero(nverts-fixed, nverts-fixed);
	Eigen::MatrixXf A_IB = Eigen::MatrixXf::Zero(nverts-fixed, fixed);
	Eigen::MatrixXf b_I = Eigen::MatrixXf::Zero(nverts - fixed, 2);
	Eigen::MatrixXf x_B = Eigen::MatrixXf::Zero(fixed, 2);



	Eigen::MatrixXf A_II_A_BB = Eigen::MatrixXf::Zero(nverts, nverts);
	for (int i = 0; i < fixed; ++i) {
	A_II_A_BB(i, i) = 1;
	}

	Eigen::MatrixXf lhsF = Eigen::MatrixXf::Zero(nverts, 3);



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

	/*Eigen::MatrixXf lhs = b_I - A_IB * x_B;
	Eigen::MatrixXf x_I = A_II.colPivHouseholderQr().solve(lhs);*/
	
	Eigen::MatrixXf x_I = A_II_A_BB.colPivHouseholderQr().solve(lhsF);

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

void TrianglesMapping::LocalGlobalParametrization(Triangles& map) {
	jacobians(map);
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
    TrianglesMapping Init(argc, argv, "arap");
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
    
	Triangles map;
	read_by_extension(Init.getOutput(), map);
	Init.LocalGlobalParametrization(map);


	for (int progress = 0; progress <= 100; ++progress) {
        updateProgressBar(progress);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << std::endl;

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
    
    
    
    /*Eigen::MatrixXf lhsF2 = Eigen::MatrixXf::Zero(nverts, 2);
    
    
    
    
    
    
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
