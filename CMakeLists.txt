project(field)
cmake_minimum_required(VERSION 3.10)

find_package(OpenMP REQUIRED)

include(ExternalProject)

ExternalProject_Add(libsleef
  GIT_REPOSITORY https://github.com/shibatch/sleef
  STAMP_DIR "${CMAKE_BINARY_DIR}/contrib/stamp/sleef"  
  BUILD_ALWAYS OFF
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib
             -DBUILD_TESTS=FALSE
             -DBUILD_DFT=FALSE
)

include_directories(${CMAKE_BINARY_DIR}/contrib/include)
link_directories(${CMAKE_BINARY_DIR}/contrib/lib)

set(CMAKE_CXX_STANDARD 14)
add_executable(field field.cpp)
add_dependencies(field libsleef)
target_link_libraries(field OpenMP::OpenMP_CXX sleef)
if(MSVC)
    target_compile_options(field INTERFACE /arch:AVX2)
else()
    target_compile_options(field INTERFACE -mavx2)
endif()