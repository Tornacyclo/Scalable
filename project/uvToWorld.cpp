#include <iostream>
#include <Eigen/Dense>
#include <ultimaille/all.h>



using namespace UM;



Matrix3d uvToWorld(const Vector3d &A, const Vector3d &B, const Vector3d &C, const Vector2d &A_prime, const Vector2d &B_prime, const Vector2d &C_prime) {
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
    Matrix3d J = uvToWorld(A, B, C, A_prime, B_prime, C_prime);

    std::cout << "Jacobian Matrix:\n" << J << std::endl;

    // Compute the inverse of the Jacobian
    Matrix3d J_inv = J.inverse();

    std::cout << "Inverse Jacobian Matrix:\n" << J_inv << std::endl;

    return 0;
}