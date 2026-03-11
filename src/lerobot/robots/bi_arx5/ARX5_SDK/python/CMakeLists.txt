cmake_minimum_required( VERSION 3.16.3 )

# ---------------------------------------------------------------------------
# Prefer conda environment packages over system packages — EXCEPT for spdlog.
# The conda PyPI spdlog package installs spdlog 1.10.0 C++ headers at
#   $CONDA_PREFIX/include/python3.12/spdlog/
# which shadow the system spdlog 1.14.1 at /usr/local/include/spdlog/.
# We deliberately do NOT prepend CONDA_PREFIX so that cmake finds the system
# spdlog 1.14.1 first via /usr/local/lib/cmake/spdlog.
# ---------------------------------------------------------------------------
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

# ---------------------------------------------------------------------------
# Use spdlog's bundled fmt (spdlog 1.14.1 bundles fmt 10.x).
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
# RPATH: embed conda lib dir so arx5_interface.so finds its deps at runtime.
# ---------------------------------------------------------------------------
set(CMAKE_BUILD_RPATH   "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ---------------------------------------------------------------------------
# Force system spdlog 1.14.1 headers (fmt::v10) to come before conda's
# python3.12/spdlog headers (fmt::v8 from PyPI spdlog 2.0.6/1.10.0).
# Pybind11 adds -isystem $CONDA_PREFIX/include/python3.12 which would
# otherwise shadow /usr/local/include/spdlog and cause an ABI mismatch.
# ---------------------------------------------------------------------------
include_directories(BEFORE /usr/local/include)

pybind11_add_module(_arx5_interface
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

target_link_libraries(_arx5_interface PRIVATE
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
target_include_directories(_arx5_interface PUBLIC ${EIGEN3_INCLUDE_DIRS})

# Hack for py310
target_include_directories(_arx5_interface PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser  $ENV{CONDA_PREFIX}/include/urdfdom_headers)

# Output to pyarx/ package directory (one level up from python/)
set_target_properties(_arx5_interface PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../pyarx")
set_target_properties(_arx5_interface PROPERTIES PREFIX "")
set_target_properties(_arx5_interface PROPERTIES OUTPUT_NAME "_arx5_interface")
