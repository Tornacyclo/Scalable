cmake_minimum_required(VERSION 3.12...3.26)

project(ultimaille-slim LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    /usr/local/include/eigen3
    ./libigl/include
)

add_executable(SLIM_intern SLIM_intern.cpp)

target_link_libraries(SLIM_intern ${CMAKE_DL_LIBS} ultimaille $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

IF (NOT WIN32)
    target_link_libraries(SLIM_intern m)
ENDIF()

# Copy asset files to build directory
file(COPY mesh_test DESTINATION .)