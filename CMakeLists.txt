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

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable
add_executable(etrobo_object_detection src/etrobo_object_detection.cpp)

# Specify libraries to link a library or executable target against
target_link_libraries(etrobo_object_detection
  ${OpenCV_LIBRARIES}
)

ament_target_dependencies(etrobo_object_detection
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
)

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