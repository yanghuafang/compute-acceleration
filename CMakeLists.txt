# 3.24 for CMAKE_CUDA_ARCHITECTURES=native and 3.25 for the workflowPresets in
# CMakePresets.json, which are what let a CI failure be reproduced with one
# local command. Everything else would build under 3.18; Ubuntu ships 3.28.
cmake_minimum_required(VERSION 3.25)

project(compute_acceleration
        VERSION 1.0.0
        DESCRIPTION "CPU and CUDA implementations of GEMM and sum reduction"
        LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CheckLanguage)
include(Sanitizers)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ACCEL_TESTS "Build the test suites" ON)
option(ACCEL_BENCHMARKS "Build the benchmark drivers" ON)
option(ACCEL_CUDA "Build CUDA targets when a toolkit is present" ON)
option(ACCEL_OPENMP "Build OpenMP targets when a runtime is found" ON)
option(ACCEL_WERROR "Treat compiler warnings as errors" OFF)
option(ACCEL_ASAN "Instrument host code with AddressSanitizer" OFF)
option(ACCEL_UBSAN "Instrument host code with UndefinedBehaviorSanitizer" OFF)
option(ACCEL_TSAN "Instrument host code with ThreadSanitizer" OFF)

# ---------------------------------------------------------------------------
# Language settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Extensions off so the code stays portable between GCC, Clang and NVCC's host
# pass rather than silently depending on a GNU extension.
set(CMAKE_CXX_EXTENSIONS OFF)

# Benchmarks are meaningless without optimisation, so an unspecified build type
# defaults to Release rather than to the empty (no -O flag) configuration.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT CMAKE_BUILD_TYPE AND NOT _is_multi_config)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
endif()

# Emitted for clangd / clang-tidy; harmless for other generators.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------------------------------------------------------------------
# Optional OpenMP runtime
# ---------------------------------------------------------------------------
set(ACCEL_HAS_OPENMP FALSE)
if(ACCEL_OPENMP)
  # AppleClang compiles OpenMP but Xcode ships no runtime, so FindOpenMP fails
  # on an otherwise healthy Mac. Homebrew's libomp is where it always lives;
  # pointing OpenMP_ROOT at it turns a confusing "not found" into a build.
  if(APPLE AND NOT DEFINED OpenMP_ROOT)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE _brew_libomp
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_brew_libomp AND EXISTS "${_brew_libomp}")
      set(OpenMP_ROOT "${_brew_libomp}")
    endif()
  endif()

  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    set(ACCEL_HAS_OPENMP TRUE)
    message(STATUS "OpenMP enabled: ${OpenMP_CXX_VERSION} "
                   "(${OpenMP_CXX_FLAGS})")
  else()
    message(STATUS "OpenMP runtime not found - building serial targets only")
  endif()
else()
  message(STATUS "OpenMP disabled by ACCEL_OPENMP=OFF")
endif()

# ---------------------------------------------------------------------------
# Optional CUDA toolchain
# ---------------------------------------------------------------------------
set(ACCEL_HAS_CUDA FALSE)
if(ACCEL_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(ACCEL_HAS_CUDA TRUE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # `native` compiles only for the GPUs actually installed, which keeps build
    # times down; override with -DCMAKE_CUDA_ARCHITECTURES=86 for a portable
    # binary or when building on a machine without a GPU.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER_VERSION} "
                   "(architectures: ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "CUDA toolkit not found - building host targets only")
  endif()
else()
  message(STATUS "CUDA disabled by ACCEL_CUDA=OFF")
endif()

accel_configure_sanitizers()

# ---------------------------------------------------------------------------
# Common warning flags
# ---------------------------------------------------------------------------
add_library(accel_warnings INTERFACE)
if(MSVC)
  target_compile_options(accel_warnings INTERFACE
                         $<$<COMPILE_LANGUAGE:CXX>:/W4>)
  if(ACCEL_WERROR)
    target_compile_options(accel_warnings INTERFACE
                           $<$<COMPILE_LANGUAGE:CXX>:/WX>)
  endif()
else()
  # -Wconversion is deliberately included: mixing int and size_t indexing
  # freely is exactly how a 4096x2048 index silently overflows.
  target_compile_options(accel_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion
                              -Wsign-conversion -Wold-style-cast
                              -Wnon-virtual-dtor -Wdouble-promotion>)
  if(ACCEL_WERROR)
    target_compile_options(accel_warnings INTERFACE
                           $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()

# ---------------------------------------------------------------------------
# Core host library
# ---------------------------------------------------------------------------
add_library(accel_core STATIC
  src/cpu/cpu_gemm.cc
  src/core/gemm_shape.cc
  src/core/matrix_utils.cc
  src/cpu/reduction.cc
)
add_library(accel::core ALIAS accel_core)

# src/ is the only include root, so every project include is written with its
# layer ("cpu/cpu_gemm.h", not "CpuGemm.hpp"). Which layer a header belongs to
# is then visible at the use site, and an unwanted edge - cuda/ reaching into
# omp/, say - is obvious in review rather than buried in a bare file name.
target_include_directories(accel_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(accel_core PUBLIC cxx_std_17)
target_link_libraries(accel_core PRIVATE accel_warnings)
accel_apply_sanitizers(accel_core)

# ---------------------------------------------------------------------------
# OpenMP kernel library
# ---------------------------------------------------------------------------
if(ACCEL_HAS_OPENMP)
  add_library(accel_omp STATIC
    src/omp/gemm_omp.cc
    src/omp/omp_control.cc
    src/omp/reduction_omp.cc
  )
  add_library(accel::omp ALIAS accel_omp)

  # OpenMP::OpenMP_CXX is PUBLIC because the flags are needed by anything that
  # links this library, not merely to build it: the runtime must appear on the
  # final link line.
  target_link_libraries(accel_omp
                        PUBLIC accel_core OpenMP::OpenMP_CXX
                        PRIVATE accel_warnings)
  accel_apply_sanitizers(accel_omp)
endif()

# ---------------------------------------------------------------------------
# CUDA kernel library
# ---------------------------------------------------------------------------
if(ACCEL_HAS_CUDA)
  add_library(accel_cuda STATIC
    src/cuda/gemm_tiled.cu
    src/cuda/sum_reduction.cu
  )
  add_library(accel::cuda ALIAS accel_cuda)

  target_link_libraries(accel_cuda PUBLIC accel_core)
  # Lets consumers link the .cu objects from plain C++ targets without needing
  # separable compilation themselves.
  set_target_properties(accel_cuda PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON
                        POSITION_INDEPENDENT_CODE ON)
  # -lineinfo makes compute-sanitizer reports point at source lines instead of
  # raw PC offsets; it does not affect generated code.
  target_compile_options(accel_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(ACCEL_BENCHMARKS)
  # Benchmark drivers include their private support headers as
  # "benchmarks/support/...", so the repository root is on their include path
  # in addition to src/, which accel_core exports.
  function(accel_add_host_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE accel_core accel_warnings)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    accel_apply_sanitizers(${name})
  endfunction()

  accel_add_host_benchmark(bench_cpu_gemm_row_major
                                benchmarks/cpu_gemm_row_major.cc)
  accel_add_host_benchmark(bench_cpu_gemm_layouts
                                benchmarks/cpu_gemm_layouts.cc)

  if(ACCEL_HAS_OPENMP)
    function(accel_add_omp_benchmark name source)
      add_executable(${name} ${source})
      target_link_libraries(${name} PRIVATE accel_omp
                                            accel_warnings)
      target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
      accel_apply_sanitizers(${name})
    endfunction()

    accel_add_omp_benchmark(bench_omp_gemm_scaling
                                    benchmarks/omp_gemm_scaling.cc)
    accel_add_omp_benchmark(bench_omp_sum_reduction
                                    benchmarks/omp_sum_reduction.cc)
  endif()

  if(ACCEL_HAS_CUDA)
    function(accel_add_cuda_benchmark name source)
      add_executable(${name} ${source})
      target_link_libraries(${name} PRIVATE accel_cuda)
      target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
      target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endfunction()

    accel_add_cuda_benchmark(bench_cuda_gemm_tiled
                                  benchmarks/cuda_gemm_tiled.cu)
    accel_add_cuda_benchmark(bench_cuda_sum_reduction
                                  benchmarks/cuda_sum_reduction.cu)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(ACCEL_TESTS)
  enable_testing()

  function(accel_add_host_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE accel_core accel_warnings)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    accel_apply_sanitizers(${name})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "host")
  endfunction()

  accel_add_host_test(test_cpu_gemm tests/cpu_gemm_test.cc)
  accel_add_host_test(test_matrix_utils tests/matrix_utils_test.cc)
  accel_add_host_test(test_cli_options tests/cli_options_test.cc)
  accel_add_host_test(test_reduction tests/reduction_test.cc)

  if(ACCEL_HAS_OPENMP)
    add_executable(test_omp_kernels tests/omp_kernels_test.cc)
    target_link_libraries(test_omp_kernels PRIVATE accel_omp
                                                   accel_warnings)
    target_include_directories(test_omp_kernels PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR})
    accel_apply_sanitizers(test_omp_kernels)
    add_test(NAME test_omp_kernels COMMAND test_omp_kernels)
    set_tests_properties(test_omp_kernels PROPERTIES LABELS "host;openmp")
  endif()

  if(ACCEL_HAS_CUDA)
    add_executable(test_cuda_kernels tests/cuda_kernels_test.cu)
    target_link_libraries(test_cuda_kernels PRIVATE accel_cuda)
    target_include_directories(test_cuda_kernels PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(test_cuda_kernels PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    add_test(NAME test_cuda_kernels COMMAND test_cuda_kernels)
    set_tests_properties(test_cuda_kernels PROPERTIES LABELS "device;cuda")
  endif()
endif()

# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "compute-acceleration ${PROJECT_VERSION}")
message(STATUS "  build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  OpenMP       : ${ACCEL_HAS_OPENMP}")
message(STATUS "  CUDA targets : ${ACCEL_HAS_CUDA}")
message(STATUS "  tests        : ${ACCEL_TESTS}")
message(STATUS "  benchmarks   : ${ACCEL_BENCHMARKS}")
message(STATUS "")
