#include <vector>
#include <array>
#include <cmath>
#include <Eigen/Dense>

struct Vector3 {
    double x, y, z;
};

struct Vector2 {
    double x, y;
};

struct Face {
    int a, b, c;
    std::string daeMaterial;
};

struct Geometry {
    std::vector<Vector3> vertices;
    std::vector<Face> faces;
    std::vector<std::vector<std::array<Vector2, 3>>> faceVertexUvs;
};

struct Node {
    Geometry geometry;
    Eigen::Matrix4d matrixWorld;
    std::vector<Node> children;
};

std::array<double, 3> ptInTriangle(const Vector2& p, const Vector2& p0, const Vector2& p1, const Vector2& p2) {
    double x0 = p.x, y0 = p.y;
    double x1 = p0.x, y1 = p0.y;
    double x2 = p1.x, y2 = p1.y;
    double x3 = p2.x, y3 = p2.y;

    double b0 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    double b1 = ((x2 - x0) * (y3 - y0) - (x3 - x0) * (y2 - y0)) / b0;
    double b2 = ((x3 - x0) * (y1 - y0) - (x1 - x0) * (y3 - y0)) / b0;
    double b3 = ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / b0;

    if (b1 > 0 && b2 > 0 && b3 > 0) {
        return {b1, b2, b3};
    } else {
        return {0, 0, 0};
    }
}

std::vector<std::array<double, 4>> annotationTest(double uvX, double uvY, const std::vector<std::array<Vector2, 3>>& faceVertexUvArray) {
    Vector2 point = {uvX, uvY};
    std::vector<std::array<double, 4>> results;

    for (size_t i = 0; i < faceVertexUvArray.size(); ++i) {
        const auto& uvs = faceVertexUvArray[i];
        auto result = ptInTriangle(point, uvs[0], uvs[1], uvs[2]);
        if (result[0] > 0 && result[1] > 0 && result[2] > 0) {
            results.push_back({static_cast<double>(i), result[0], result[1], result[2]});
        }
    }

    return results;
}

Vector3 traversePolygonsForGeometries(Node& node, double uvx, double uvy) {
    if (!node.geometry.vertices.empty() && !node.geometry.faces.empty()) {
        auto baryData = annotationTest(uvx, uvy, node.geometry.faceVertexUvs[0]);
        if (!baryData.empty()) {
            for (const auto& data : baryData) {
                int faceIndex = static_cast<int>(data[0]);
                const auto& face = node.geometry.faces[faceIndex];

                if (face.daeMaterial == "frontMaterial" || face.daeMaterial == "print_area1_0") {
                    const auto& vertexa = node.geometry.vertices[face.a];
                    const auto& vertexb = node.geometry.vertices[face.b];
                    const auto& vertexc = node.geometry.vertices[face.c];

                    double worldX = vertexa.x * data[1] + vertexb.x * data[2] + vertexc.x * data[3];
                    double worldY = vertexa.y * data[1] + vertexb.y * data[2] + vertexc.y * data[3];
                    double worldZ = vertexa.z * data[1] + vertexb.z * data[2] + vertexc.z * data[3];

                    Eigen::Vector4d localPoint(worldX, worldY, worldZ, 1.0);
                    Eigen::Vector4d worldPoint = node.matrixWorld * localPoint;

                    return {worldPoint.x(), worldPoint.y(), worldPoint.z()};
                }
            }
        }
    }

    for (auto& child : node.children) {
        auto result = traversePolygonsForGeometries(child, uvx, uvy);
        if (result.x != 0 || result.y != 0 || result.z != 0) {
            return result;
        }
    }

    return {0, 0, 0}; // Default return value if no match found
}



int main() {
    // Create and initialize your nodes, geometries, faces, and matrices here.

    Node rootNode; // Assume this is initialized properly
    double uvx = 0.5; // Example UV coordinate
    double uvy = 0.5;

    Vector3 worldVectorPoint = traversePolygonsForGeometries(rootNode, uvx, uvy);

    // Print or use the result
    std::cout << "World Vector Point: (" << worldVectorPoint.x << ", " << worldVectorPoint.y << ", " << worldVectorPoint.z << ")\n";

    return 0;
}