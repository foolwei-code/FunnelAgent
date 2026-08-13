cmake_minimum_required(VERSION 3.16)

# ============================================================================
# Project definition
# ============================================================================

project(FunnelRAG-Reranker
    VERSION 1.0.0
    DESCRIPTION "FunnelRAG Cross-Encoder Reranker — ONNX Runtime + gRPC"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# CMake options
# ============================================================================

option(BUILD_TESTS      "Build unit tests"       OFF)
option(BUILD_BENCHMARKS "Build benchmark binary"  OFF)
option(BUILD_SHARED_LIB "Build shared library instead of static" OFF)
option(ENABLE_LTO      "Enable link-time optimization" OFF)
option(ENABLE_ASAN     "Enable AddressSanitizer" OFF)

# ============================================================================
# Compiler warnings and flags
# ============================================================================

add_library(project_warnings INTERFACE)

target_compile_options(project_warnings INTERFACE
    -Wall
    -Wextra
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    # -Werror  # Uncomment to treat warnings as errors in CI
)

# Optional: Link-Time Optimization
if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO enabled")
    else()
        message(WARNING "LTO not supported: ${ipo_output}")
    endif()
endif()

# Optional: AddressSanitizer
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
    message(STATUS "AddressSanitizer enabled")
endif()

# ============================================================================
# Dependency finding
# ============================================================================

# --- Protobuf ---
find_package(protobuf REQUIRED)

# --- gRPC ---
find_package(gRPC REQUIRED)

# --- ONNX Runtime ---
# First try the CMake config, then fall back to our Find module
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(onnxruntime REQUIRED)
endif()

message(STATUS "Protobuf version: ${Protobuf_VERSION}")
message(STATUS "gRPC found: YES")
message(STATUS "ONNX Runtime includes: ${onnxruntime_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime libraries: ${onnxruntime_LIBRARIES}")

# ============================================================================
# Protobuf / gRPC code generation
# ============================================================================

set(PROTO_SRC_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../proto")
set(PROTO_GEN_DIR  "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

set(PROTO_FILE "${PROTO_SRC_DIR}/reranker.proto")

add_custom_command(
    OUTPUT
        "${PROTO_GEN_DIR}/reranker.pb.cc"
        "${PROTO_GEN_DIR}/reranker.pb.h"
        "${PROTO_GEN_DIR}/reranker.grpc.pb.cc"
        "${PROTO_GEN_DIR}/reranker.grpc.pb.h"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        --proto_path=${PROTO_SRC_DIR}
        --cpp_out=${PROTO_GEN_DIR}
        --grpc_out=${PROTO_GEN_DIR}
        --plugin=protoc-gen-grpc=${gRPC_CPP_PLUGIN}
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf/gRPC sources from reranker.proto"
)

# Custom target so other targets can depend on the generated files
add_custom_target(generate_proto ALL
    DEPENDS
        "${PROTO_GEN_DIR}/reranker.pb.cc"
        "${PROTO_GEN_DIR}/reranker.pb.h"
        "${PROTO_GEN_DIR}/reranker.grpc.pb.cc"
        "${PROTO_GEN_DIR}/reranker.grpc.pb.h"
)

# ============================================================================
# Main reranker server executable
# ============================================================================

add_executable(reranker_server
    src/onnx_reranker.cpp
    src/grpc_server.cpp
    "${PROTO_GEN_DIR}/reranker.pb.cc"
    "${PROTO_GEN_DIR}/reranker.grpc.pb.cc"
)

add_dependencies(reranker_server generate_proto)

target_compile_features(reranker_server PRIVATE cxx_std_17)

target_link_libraries(reranker_server PRIVATE
    project_warnings
    gRPC::grpc++
    protobuf::libprotobuf
    ${onnxruntime_LIBRARIES}
)

target_include_directories(reranker_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROTO_GEN_DIR}
    ${onnxruntime_INCLUDE_DIRS}
)

# Set output name and version defines
target_compile_definitions(reranker_server PRIVATE
    RERANKER_VERSION="${PROJECT_VERSION}"
    RERANKER_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    RERANKER_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    RERANKER_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# ============================================================================
# Optional: Unit tests
# ============================================================================

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(reranker_test
        tests/test_onnx_reranker.cpp
        src/onnx_reranker.cpp
    )

    target_link_libraries(reranker_test PRIVATE
        project_warnings
        GTest::gtest
        GTest::gtest_main
        ${onnxruntime_LIBRARIES}
    )

    target_include_directories(reranker_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${onnxruntime_INCLUDE_DIRS}
    )

    target_compile_features(reranker_test PRIVATE cxx_std_17)

    add_test(NAME reranker_test COMMAND reranker_test)
    message(STATUS "Unit tests enabled")
endif()

# ============================================================================
# Optional: Benchmarks
# ============================================================================

if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)

    add_executable(reranker_bench
        benchmarks/bench_onnx_reranker.cpp
        src/onnx_reranker.cpp
    )

    target_link_libraries(reranker_bench PRIVATE
        project_warnings
        benchmark::benchmark
        benchmark::benchmark_main
        ${onnxruntime_LIBRARIES}
    )

    target_include_directories(reranker_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${onnxruntime_INCLUDE_DIRS}
    )

    target_compile_features(reranker_bench PRIVATE cxx_std_17)
    message(STATUS "Benchmarks enabled")
endif()

# ============================================================================
# Install targets
# ============================================================================

include(GNUInstallDirs)

install(TARGETS reranker_server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_reranker.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/funnelrag
)

# Install CMake Find module for onnxruntime (if we have one)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Findonnxruntime.cmake")
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Findonnxruntime.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== ${PROJECT_NAME} v${PROJECT_VERSION} ===")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests:      ${BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  ASAN:             ${ENABLE_ASAN}")
message(STATUS "  LTO:              ${ENABLE_LTO}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
