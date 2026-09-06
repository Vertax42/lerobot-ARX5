cmake_minimum_required(VERSION 3.14)

project(XenseVR_PC_Service_PYBIND11 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_host_system_information(RESULT ISA_NAME QUERY OS_PLATFORM) # Added: Important for UNIX specific logic
message(STATUS "OS_PLATFORM (ISA_NAME): ${ISA_NAME}")

include(GNUInstallDirs) # Add this line
find_package(pybind11 REQUIRED)

# py_bindings.cpp parses the device state JSON with nlohmann. This used to
# resolve by accident: setup_env.sh copied a vendored nlohmann/ out of the
# XenseVR-PC-Service submodule into include/ alongside the SDK header. That
# submodule is gone — the SDK now comes from the installed .deb, which does not
# carry nlohmann — so the dependency has to be declared and comes from conda.
find_package(nlohmann_json CONFIG REQUIRED)

# Python Bindings for py_bindings.cpp
pybind11_add_module(xensevr_pc_service_sdk MODULE bindings/py_bindings.cpp)

# Link xensevr_pc_service_sdk module against pybind11
target_link_libraries(xensevr_pc_service_sdk PRIVATE pybind11::module nlohmann_json::nlohmann_json)

# Add include directories and link libraries for PXREARobotSDK to xensevr_pc_service_sdk target
if(WIN32)
  target_include_directories(xensevr_pc_service_sdk PUBLIC
      ${PROJECT_SOURCE_DIR}/include
  )
  target_link_directories(xensevr_pc_service_sdk PUBLIC ${PROJECT_SOURCE_DIR}/lib)
  target_link_libraries(xensevr_pc_service_sdk PUBLIC
      PXREARobotSDK.dll # Assuming this is how PXREARobotSDK is linked, mirroring ConsoleDemo
  )
endif()

if(UNIX)
  # No per-architecture subdirectory: setup_env.sh picks SDK/x64 or SDK/arm64
  # out of the installed .deb and stages the result flat into include/ and
  # lib/, so what lands here is already the right build for this host. The
  # aarch64 branch that used to live here looked in include/aarch64, which
  # nothing creates any more.
  target_include_directories(xensevr_pc_service_sdk PUBLIC
      ${PROJECT_SOURCE_DIR}/include
  )
  target_link_directories(xensevr_pc_service_sdk PUBLIC ${PROJECT_SOURCE_DIR}/lib)
  target_link_libraries(xensevr_pc_service_sdk PUBLIC
      PXREARobotSDK
  )
endif()

# Install the Python module
install(TARGETS xensevr_pc_service_sdk
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Installs to <prefix>/lib
                                                # You might want a Python-specific path like:
                                                # DESTINATION lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
)
