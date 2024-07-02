#include "uvToWorld.h"
#include <iostream>



Eigen::Matrix3d uvToWorld(const vec3 &A, const vec3 &B, const vec3 &C, const vec2 &A_prime, const vec2 &B_prime, const vec2 &C_prime) {
    Eigen::Matrix3d J_f;

    // Compute the edges in 3D
    vec3 AB = B - A;
    vec3 AC = C - A;

    // Compute the edges in 2D
    vec2 AB_prime = B_prime - A_prime;
    vec2 AC_prime = C_prime - A_prime;


    // Set up the Jacobian matrix, dim 0 = x, dim 2 = y, dim 1 = z
    J_f(0, 0) = AB_prime[0] / AB[0];
    J_f(0, 1) = AB_prime[0] / AB[2];
    J_f(0, 2) = AB_prime[0] / AB[1];

    J_f(1, 0) = AB_prime[2] / AB[0];
    J_f(1, 1) = AB_prime[2] / AB[2];
    J_f(1, 2) = AB_prime[2] / AB[1];

    J_f(2, 0) = AC_prime[0] / AC[0];
    J_f(2, 1) = AC_prime[0] / AC[2];
    J_f(2, 2) = AC_prime[0] / AC[1];


    return J_f;
}



int main() {
    // Define the vertices of the 3D triangle
    vec3 A(0.0, 0.0, 0.0);
    vec3 B(1.0, 0.0, 0.0);
    vec3 C(0.0, 1.0, 0.0);

    // Define the vertices of the 2D triangle (deformation)
    vec2 A_prime(0.0, 0.0);
    vec2 B_prime(1.0, 0.0);
    vec2 C_prime(0.0, 1.0);

    // Compute the Jacobian
    Eigen::Matrix3d J_f = uvToWorld(A, B, C, A_prime, B_prime, C_prime);

    std::cout << "Jacobian Matrix:\n" << J_f << std::endl;

    // Compute the inverse of the Jacobian
    Eigen::Matrix3d J_f_inv = J_f.inverse();

    std::cout << "Inverse Jacobian Matrix:\n" << J_f_inv << std::endl;

    return 0;
}