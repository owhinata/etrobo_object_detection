cmake_minimum_required(VERSION 3.8)
project(etrobo_object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# Download and extract ONNX Runtime at build time
include(FetchContent)

set(ONNXRUNTIME_VERSION "1.22.0")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")

FetchContent_Declare(
  onnxruntime
  URL ${ONNXRUNTIME_URL}
)

FetchContent_MakeAvailable(onnxruntime)

# Set ONNX Runtime paths
set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)

# Include ONNX Runtime headers
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable (ONNX Runtime version)
add_executable(etrobo_object_detection src/etrobo_object_detection_onnx.cpp)

# Specify libraries to link a library or executable target against
target_link_libraries(etrobo_object_detection
  ${OpenCV_LIBRARIES}
  onnxruntime
)

# Set rpath for runtime library loading
set_target_properties(etrobo_object_detection PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

ament_target_dependencies(etrobo_object_detection
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
)

# Install ONNX Runtime library
install(FILES ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.${ONNXRUNTIME_VERSION}
        DESTINATION lib)
install(FILES ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
        DESTINATION lib)

# Install targets
install(TARGETS
  etrobo_object_detection
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()