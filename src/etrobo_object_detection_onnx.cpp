#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>

class ObjectDetectionNode : public rclcpp::Node {
public:
  ObjectDetectionNode() : Node("object_detection") {
    try {
      // Initialize ONNX Runtime
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ObjectDetection");
      
      // Create session options
      session_options_ = std::make_unique<Ort::SessionOptions>();
      session_options_->SetIntraOpNumThreads(1);
      session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
      
      // Load YOLO model
      const char* model_path = "yolov8n.onnx";
      session_ = std::make_unique<Ort::Session>(*env_, model_path, *session_options_);
      
      RCLCPP_INFO(this->get_logger(), "ONNX Runtime YOLO model loaded successfully");
      
      // Get input/output info
      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      memory_info_ = std::make_unique<Ort::MemoryInfo>(std::move(memory_info));
      
      // Print model input/output information
      size_t num_input_nodes = session_->GetInputCount();
      size_t num_output_nodes = session_->GetOutputCount();
      
      RCLCPP_INFO(this->get_logger(), "=== MODEL INFO ===");
      RCLCPP_INFO(this->get_logger(), "Number of inputs: %zu", num_input_nodes);
      RCLCPP_INFO(this->get_logger(), "Number of outputs: %zu", num_output_nodes);
      
      // Get input name and shape
      auto input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
      input_names_.push_back(std::string(input_name_ptr.get()));
      input_node_names_.push_back(input_names_.back().c_str());
      
      auto input_type_info = session_->GetInputTypeInfo(0);
      auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
      auto input_dims = input_tensor_info.GetShape();
      
      RCLCPP_INFO(this->get_logger(), "=== INPUT TENSOR INFO ===");
      RCLCPP_INFO(this->get_logger(), "Input name: %s", input_names_.back().c_str());
      RCLCPP_INFO(this->get_logger(), "Input dimensions: %zu", input_dims.size());
      std::string input_shape = "Input shape: [";
      for (size_t i = 0; i < input_dims.size(); i++) {
        if (i > 0) input_shape += ", ";
        input_shape += std::to_string(input_dims[i]);
      }
      input_shape += "]";
      RCLCPP_INFO(this->get_logger(), "%s", input_shape.c_str());
      
      // Calculate input elements
      size_t input_elements = 1;
      for (size_t i = 0; i < input_dims.size(); ++i) {
        input_elements *= input_dims[i];
      }
      RCLCPP_INFO(this->get_logger(), "Input tensor elements: %zu", input_elements);
      
      // Get output name and shape
      auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator_);
      output_names_.push_back(std::string(output_name_ptr.get()));
      output_node_names_.push_back(output_names_.back().c_str());
      
      auto output_type_info = session_->GetOutputTypeInfo(0);
      auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
      auto output_dims = output_tensor_info.GetShape();
      
      RCLCPP_INFO(this->get_logger(), "=== OUTPUT TENSOR INFO (STATIC) ===");
      RCLCPP_INFO(this->get_logger(), "Output name: %s", output_names_.back().c_str());
      RCLCPP_INFO(this->get_logger(), "Output dimensions: %zu", output_dims.size());
      std::string output_shape = "Output shape: [";
      for (size_t i = 0; i < output_dims.size(); i++) {
        if (i > 0) output_shape += ", ";
        output_shape += std::to_string(output_dims[i]);
      }
      output_shape += "]";
      RCLCPP_INFO(this->get_logger(), "%s", output_shape.c_str());
      
      // Calculate output elements
      size_t output_elements = 1;
      for (size_t i = 0; i < output_dims.size(); ++i) {
        output_elements *= output_dims[i];
      }
      RCLCPP_INFO(this->get_logger(), "Output tensor elements: %zu", output_elements);
      RCLCPP_INFO(this->get_logger(), "==================");
      
    } catch (const Ort::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "ONNX Runtime error: %s", e.what());
      return;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error loading model: %s", e.what());
      return;
    }

    // Create subscription
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.best_effort();

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", qos,
        std::bind(&ObjectDetectionNode::image_callback, this,
                  std::placeholders::_1));
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      // Convert ROS image to OpenCV format
      cv_bridge::CvImagePtr cv_ptr =
          cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat img = cv_ptr->image;

      if (!session_) {
        RCLCPP_WARN(this->get_logger(), "Model not loaded, skipping detection");
        return;
      }

      // Perform object detection
      cv::Mat result_img = perform_detection(img);

      // Display result
      cv::imshow("result", result_img);
      cv::waitKey(1);

    } catch (cv_bridge::Exception &e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  cv::Mat perform_detection(const cv::Mat &image) {
    cv::Mat result = image.clone();

    try {
      // Preprocess image
      cv::Mat blob;
      cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(640, 640),
                             cv::Scalar(0, 0, 0), true, false);

      // Prepare input tensor
      std::vector<int64_t> input_shape = {1, 3, 640, 640};
      size_t input_tensor_size = 1 * 3 * 640 * 640;
      
      auto input_tensor = Ort::Value::CreateTensor<float>(
          *memory_info_, (float*)blob.data, input_tensor_size,
          input_shape.data(), input_shape.size());

      // Run inference
      RCLCPP_INFO(this->get_logger(), "Running ONNX inference...");
      auto output_tensors = session_->Run(
          Ort::RunOptions{nullptr},
          input_node_names_.data(), &input_tensor, 1,
          output_node_names_.data(), 1);

      RCLCPP_INFO(this->get_logger(), "ONNX inference completed successfully");

      // Process output
      if (!output_tensors.empty()) {
        auto& output_tensor = output_tensors[0];
        auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        
        // Detailed output tensor information
        RCLCPP_INFO(this->get_logger(), "=== OUTPUT TENSOR INFO ===");
        RCLCPP_INFO(this->get_logger(), "Output tensor dimensions: %zu", shape.size());
        std::string shape_str = "Output tensor shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
          if (i > 0) shape_str += ", ";
          shape_str += std::to_string(shape[i]);
        }
        shape_str += "]";
        RCLCPP_INFO(this->get_logger(), "%s", shape_str.c_str());
        
        // Calculate total elements
        size_t total_elements = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
          total_elements *= shape[i];
        }
        RCLCPP_INFO(this->get_logger(), "Total tensor elements: %zu", total_elements);
        
        float* output_data = output_tensor.GetTensorMutableData<float>();
        
        // Show first 10 values to understand data layout
        RCLCPP_INFO(this->get_logger(), "First 10 raw values: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f",
                    output_data[0], output_data[1], output_data[2], output_data[3], output_data[4],
                    output_data[5], output_data[6], output_data[7], output_data[8], output_data[9]);
                    
        RCLCPP_INFO(this->get_logger(), "========================");
        
        process_yolo_output(result, output_data, shape, image.cols, image.rows);
      }

    } catch (const Ort::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "ONNX Runtime inference error: %s", e.what());
    }

    return result;
  }

  void process_yolo_output(cv::Mat& image, float* output_data, 
                          const std::vector<int64_t>& shape,
                          int img_width, int img_height) {
    const float confidence_threshold = 0.5;
    const float nms_threshold = 0.4;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // YOLOv8 output shape: [1, 84, 8400]
    // 84 = 4 (bbox coords) + 80 (class scores)
    int num_detections = shape[2]; // 8400
    int num_features = shape[1];   // 84

    for (int i = 0; i < num_detections; ++i) {
      // Get detection data for detection i
      float* detection = output_data + i * num_features;
      
      // Extract bounding box coordinates (first 4 elements)
      float cx = detection[0];
      float cy = detection[1];
      float w = detection[2];
      float h = detection[3];

      // Find class with highest confidence (elements 4-83)
      float max_confidence = 0.0f;
      int class_id = -1;
      for (int j = 4; j < num_features; ++j) {
        if (detection[j] > max_confidence) {
          max_confidence = detection[j];
          class_id = j - 4;
        }
      }

      if (max_confidence > confidence_threshold) {
        // Convert to actual image coordinates
        int x = static_cast<int>((cx - w / 2) * img_width / 640);
        int y = static_cast<int>((cy - h / 2) * img_height / 640);
        int width = static_cast<int>(w * img_width / 640);
        int height = static_cast<int>(h * img_height / 640);

        boxes.emplace_back(x, y, width, height);
        confidences.push_back(max_confidence);
        class_ids.push_back(class_id);
      }
    }

    // Apply non-maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold,
                      indices);

    // Draw bounding boxes
    for (int idx : indices) {
      cv::Rect box = boxes[idx];
      float confidence = confidences[idx];
      int class_id = class_ids[idx];

      // Draw rectangle
      cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

      // Draw label
      std::string label = "Class " + std::to_string(class_id) + ": " +
                          std::to_string(static_cast<int>(confidence * 100)) +
                          "%";
      int baseline;
      cv::Size label_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

      cv::rectangle(image, cv::Point(box.x, box.y - label_size.height - 10),
                    cv::Point(box.x + label_size.width, box.y),
                    cv::Scalar(0, 255, 0), -1);

      cv::putText(image, label, cv::Point(box.x, box.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  
  // ONNX Runtime components
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;
  Ort::AllocatorWithDefaultOptions allocator_;
  
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_node_names_;
  std::vector<const char*> output_node_names_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObjectDetectionNode>();

  try {
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node->get_logger(), "Exception in main: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}