cmake_minimum_required(VERSION 3.15)

# Require standard C++17 for all targets.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(asmodean CXX)

# Default to a Release build when nothing is specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Release, RelWithDebInfo, Debug or MinSizeRel)."
      FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Compiled extension is dropped straight into the importable python package.
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/asmodean")

set(WARNING_FLAGS
    $<$<CXX_COMPILER_ID:GNU>:
        -Wall -Wextra -Wpedantic
    >
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -Wall -Wextra -Wpedantic -Wno-missing-braces
    >
)

# --- Dependencies -----------------------------------------------------------
# Boost (header-only: numeric/odeint here, plus the math special functions the
# lanfear SCF headers use). BOOST_ROOT is exported by the `boost` module that
# `load_py313` restores.
find_package(Boost 1.70 REQUIRED)

# OpenMP for batches of independent scattering experiments.
find_package(OpenMP)

# Locate pybind11 via the interpreter in the active environment, so the .so is
# built for the exact python that will import it.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_status)
if(NOT _pybind11_status EQUAL 0)
  message(FATAL_ERROR "Could not locate pybind11 via ${Python3_EXECUTABLE}. "
                      "Did you run load_py313 first?")
endif()
find_package(pybind11 CONFIG REQUIRED)

# lanfear's C++ core is header-only; asmodean compiles the analytical potentials
# (SCF, disc, composite) straight from its headers. Search order: an explicit
# -DLANFEAR_INCLUDE_DIR, then $LANFEAR_ROOT/include, then the source checkout
# of the lanfear package importable by the active interpreter.
set(LANFEAR_INCLUDE_DIR "" CACHE PATH "Directory containing lanfear/*.hpp")
if(NOT LANFEAR_INCLUDE_DIR AND DEFINED ENV{LANFEAR_ROOT})
  set(LANFEAR_INCLUDE_DIR "$ENV{LANFEAR_ROOT}/include")
endif()
if(NOT LANFEAR_INCLUDE_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import os, lanfear; print(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(lanfear.__file__))), 'include'))"
    OUTPUT_VARIABLE _lanfear_guess
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _lanfear_status
    ERROR_QUIET)
  if(_lanfear_status EQUAL 0)
    set(LANFEAR_INCLUDE_DIR "${_lanfear_guess}")
  endif()
endif()
if(NOT EXISTS "${LANFEAR_INCLUDE_DIR}/lanfear/composite_potential.hpp")
  message(FATAL_ERROR "Could not find the lanfear headers (looked in "
                      "'${LANFEAR_INCLUDE_DIR}'). Set -DLANFEAR_INCLUDE_DIR=... or "
                      "export LANFEAR_ROOT=/path/to/lanfear.")
endif()
message(STATUS "Using lanfear headers from ${LANFEAR_INCLUDE_DIR}")

# --- Targets ----------------------------------------------------------------
add_library(asmodean_core INTERFACE)
target_include_directories(asmodean_core INTERFACE
  "${PROJECT_SOURCE_DIR}/include")
# lanfear and Boost as SYSTEM includes so their warnings don't trip -Wpedantic.
target_include_directories(asmodean_core SYSTEM INTERFACE
  "${LANFEAR_INCLUDE_DIR}" ${Boost_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
  target_link_libraries(asmodean_core INTERFACE OpenMP::OpenMP_CXX)
endif()

# Standalone test harness (pure C++, no python) for quick physics checks.
add_executable(test_core "src/test_core.cpp")
target_link_libraries(test_core PRIVATE asmodean_core)
target_compile_options(test_core PRIVATE ${WARNING_FLAGS})

# Python extension module -> asmodean/_core.*.so
pybind11_add_module(_core "src/python_bindings.cpp")
target_link_libraries(_core PRIVATE asmodean_core)
target_compile_options(_core PRIVATE ${WARNING_FLAGS})
