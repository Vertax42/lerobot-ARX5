cmake_minimum_required( VERSION 3.16.3 )

# ---------------------------------------------------------------------------
# Prefer conda environment packages over system packages.
# ---------------------------------------------------------------------------
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

# ---------------------------------------------------------------------------
# Use spdlog's bundled fmt (spdlog 1.14.1 bundles fmt 11.x).
# Prevents ABI mismatch with any separately installed system libfmt.
# MUST be set before find_package(spdlog).
# ---------------------------------------------------------------------------
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "Use spdlog bundled fmt, not system fmt" FORCE)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(spdlog REQUIRED)

# ---------------------------------------------------------------------------
# RPATH: embed conda lib dir so arx5_interface.so finds spdlog at runtime.
# ---------------------------------------------------------------------------
set(CMAKE_BUILD_RPATH   "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

pybind11_add_module(arx5_interface
arx5_pybind.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../src/app/joint_controller.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../src/app/cartesian_controller.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../src/app/controller_base.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../src/utils.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/libcan)
add_compile_options(-pthread)

# Add conda lib directory to link directories
link_directories($ENV{CONDA_PREFIX}/lib)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/aarch64)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/x86_64)
endif()

target_link_libraries(arx5_interface PRIVATE
${LIB_DIR}/libhardware.so
${LIB_DIR}/libsolver.so
    spdlog::spdlog
    Eigen3::Eigen
    Threads::Threads
    kdl_parser::kdl_parser
    orocos-kdl
    pthread
    soem
)
target_include_directories(arx5_interface PUBLIC ${EIGEN3_INCLUDE_DIRS})

# Hack for py310
target_include_directories(arx5_interface PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser  $ENV{CONDA_PREFIX}/include/urdfdom_headers)

# Optional: set the output directory for the built module
set_target_properties(arx5_interface PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(arx5_interface PROPERTIES PREFIX "")
set_target_properties(arx5_interface PROPERTIES OUTPUT_NAME "arx5_interface")
