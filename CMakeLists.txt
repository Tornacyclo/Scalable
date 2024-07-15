cmake_minimum_required(VERSION 3.12...3.26)

project(sandbox LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT (CMAKE_BUILD_TYPE OR DEFINED ENV{CMAKE_BUILD_TYPE}))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release default")
endif()

find_package(OpenMP COMPONENTS CXX)

if (MSVC)
    # warning level 4 (and all warnings as errors, /WX)
    add_compile_options(/W4)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic)
endif()

include(FetchContent)

# Download the JSON file with the latest release info
file(DOWNLOAD
  "https://api.github.com/repos/ssloy/ultimaille/releases/latest"
  "${CMAKE_BINARY_DIR}/ultimaille_latest_release.json"
  TIMEOUT 60
  STATUS status
  LOG log
)

list(GET status 0 status_code)
if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "Error downloading latest release info: ${log}")
endif()

# Read the tag_name from the JSON file
file(READ "${CMAKE_BINARY_DIR}/ultimaille_latest_release.json" latest_release_json)
string(REGEX MATCH "\"tag_name\": \"([^\"]+)\"" _ ${latest_release_json})
set(latest_tag ${CMAKE_MATCH_1})

if(NOT latest_tag)
  message(FATAL_ERROR "Error parsing latest release tag from JSON: ${latest_release_json}")
endif()

message(STATUS "Latest release tag: ${latest_tag}")

FetchContent_Declare(
  ultimaille
  GIT_REPOSITORY https://github.com/ssloy/ultimaille
  GIT_TAG        ${latest_tag}
)
FetchContent_MakeAvailable(ultimaille)
include_directories(${ultimaille_SOURCE_DIR})

# Define repository information
set(GITHUB_REPO "BrunoLevy/GraphiteThree")
set(ZIP_FILE "${CMAKE_SOURCE_DIR}/graphite.zip")
set(EXTRACT_DIR "${CMAKE_BINARY_DIR}")

# Get the latest release data using curl
execute_process(
    COMMAND curl -s https://api.github.com/repos/${GITHUB_REPO}/releases/latest
    OUTPUT_VARIABLE RELEASE_DATA
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if the release data was retrieved successfully
if(NOT RELEASE_DATA)
    message(FATAL_ERROR "Failed to get the latest release data.")
endif()

# Debug output to check the fetched JSON
message(STATUS "Release data: ${RELEASE_DATA}")

# Write the release data to a temporary file for jq to process
file(WRITE "${CMAKE_BINARY_DIR}/release_data.json" "${RELEASE_DATA}")

# Use jq to extract the download URL for the Windows 64-bit zip file
execute_process(
    COMMAND jq -r ".assets[] | select(.name | endswith(\"win64.zip\")) | .browser_download_url" ${CMAKE_BINARY_DIR}/release_data.json
    OUTPUT_VARIABLE LATEST_RELEASE_URL
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if the URL was retrieved successfully
if(NOT LATEST_RELEASE_URL)
    message(FATAL_ERROR "Failed to get the latest release URL.")
endif()

# Debug output to check the extracted URL
message(STATUS "Latest release URL: ${LATEST_RELEASE_URL}")

# Download the latest release zip file
file(DOWNLOAD ${LATEST_RELEASE_URL} ${ZIP_FILE})

# Unzip the downloaded file
execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${ZIP_FILE}
    WORKING_DIRECTORY ${EXTRACT_DIR}
)

# Remove the downloaded zip file
file(REMOVE_RECURSE ${ZIP_FILE})

# Clean up the temporary JSON file
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/release_data.json)

add_subdirectory(project)