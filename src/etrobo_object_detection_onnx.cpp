#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <iomanip>
#include <map>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <set>
#include <sstream>
#include <vector>
#include <vision_msgs/msg/bounding_box2_d.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

class ObjectDetectionNode : public rclcpp::Node {
public:
  ObjectDetectionNode() : Node("object_detection") {
    // Initialize COCO class names
    initializeCocoLabels();
    // Declare ROS2 parameters with default values
    declare_parameters();

    try {
      initialize_onnx_runtime();
      setup_input_output_names();
      log_model_info();

    } catch (const Ort::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "ONNX Runtime error: %s", e.what());
      return;
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error loading model: %s", e.what());
      return;
    }

    setup_subscription();
    setup_publisher();
  }

private:
  void initializeCocoLabels() {
    coco_labels_ = {"person",        "bicycle",      "car",
                    "motorcycle",    "airplane",     "bus",
                    "train",         "truck",        "boat",
                    "traffic light", "fire hydrant", "stop sign",
                    "parking meter", "bench",        "bird",
                    "cat",           "dog",          "horse",
                    "sheep",         "cow",          "elephant",
                    "bear",          "zebra",        "giraffe",
                    "backpack",      "umbrella",     "handbag",
                    "tie",           "suitcase",     "frisbee",
                    "skis",          "snowboard",    "sports ball",
                    "kite",          "baseball bat", "baseball glove",
                    "skateboard",    "surfboard",    "tennis racket",
                    "bottle",        "wine glass",   "cup",
                    "fork",          "knife",        "spoon",
                    "bowl",          "banana",       "apple",
                    "sandwich",      "orange",       "broccoli",
                    "carrot",        "hot dog",      "pizza",
                    "donut",         "cake",         "chair",
                    "couch",         "potted plant", "bed",
                    "dining table",  "toilet",       "tv",
                    "laptop",        "mouse",        "remote",
                    "keyboard",      "cell phone",   "microwave",
                    "oven",          "toaster",      "sink",
                    "refrigerator",  "book",         "clock",
                    "vase",          "scissors",     "teddy bear",
                    "hair drier",    "toothbrush"};
  }

  std::string getClassName(int class_id) {
    if (class_id >= 0 && class_id < static_cast<int>(coco_labels_.size())) {
      return coco_labels_[class_id];
    }
    return "unknown";
  }

  void declare_parameters() {
    // Model parameters
    this->declare_parameter("model_path", "yolov8n.onnx");

    // Inference parameters
    this->declare_parameter("confidence_threshold", 0.5);
    this->declare_parameter("nms_threshold", 0.4);
    this->declare_parameter("target_classes",
                            std::vector<int64_t>{39}); // bottle only

    // Runtime parameters
    this->declare_parameter("num_threads", 2);

    // I/O parameters
    this->declare_parameter("input_topic", "/image_raw");
    this->declare_parameter("output_topic", "/object_detection");

    // Get model parameters
    model_path_ = this->get_parameter("model_path").as_string();

    // Get inference parameters
    confidence_threshold_ =
        this->get_parameter("confidence_threshold").as_double();
    nms_threshold_ = this->get_parameter("nms_threshold").as_double();
    auto target_classes_param =
        this->get_parameter("target_classes").as_integer_array();
    target_classes_.clear();
    for (auto class_id : target_classes_param) {
      target_classes_.insert(static_cast<int>(class_id));
    }

    // Get runtime parameters
    num_threads_ = this->get_parameter("num_threads").as_int();

    // Get I/O parameters
    input_topic_ = this->get_parameter("input_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();

    // Log parameter values
    RCLCPP_INFO(this->get_logger(), "Parameters:");
    // Model parameters
    RCLCPP_INFO(this->get_logger(), "  model_path: %s", model_path_.c_str());

    // Inference parameters
    RCLCPP_INFO(this->get_logger(), "  confidence_threshold: %.2f",
                confidence_threshold_);
    RCLCPP_INFO(this->get_logger(), "  nms_threshold: %.2f", nms_threshold_);

    // Runtime parameters
    RCLCPP_INFO(this->get_logger(), "  num_threads: %d", num_threads_);

    // I/O parameters
    RCLCPP_INFO(this->get_logger(), "  input_topic: %s", input_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  output_topic: %s",
                output_topic_.c_str());

    // Log target classes
    std::stringstream target_classes_str;
    bool first = true;
    for (int class_id : target_classes_) {
      if (!first)
        target_classes_str << ", ";
      target_classes_str << class_id << "(" << getClassName(class_id) << ")";
      first = false;
    }
    RCLCPP_INFO(this->get_logger(), "  target_classes: [%s]",
                target_classes_str.str().c_str());
  }

  void setup_subscription() {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.best_effort();

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        input_topic_, qos,
        std::bind(&ObjectDetectionNode::image_callback, this,
                  std::placeholders::_1));
  }

  void setup_publisher() {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.best_effort();

    image_publisher_ =
        this->create_publisher<sensor_msgs::msg::CompressedImage>(
            output_topic_ + "/image/compressed", qos);

    detection_publisher_ =
        this->create_publisher<vision_msgs::msg::Detection2DArray>(
            output_topic_ + "/detections", qos);
  }

  void initialize_onnx_runtime() {
    // Initialize ONNX Runtime
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      "ObjectDetection");

    // Create session options
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(num_threads_);
    session_options_->SetInterOpNumThreads(num_threads_);
    session_options_->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options_->SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load YOLO model
    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(),
                                              *session_options_);

    RCLCPP_INFO(this->get_logger(),
                "ONNX Runtime YOLO model loaded successfully");

    // Get input/output info
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    memory_info_ = std::make_unique<Ort::MemoryInfo>(std::move(memory_info));
  }

  void setup_input_output_names() {
    // Get input name
    auto input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
    input_names_.push_back(std::string(input_name_ptr.get()));
    input_node_names_.push_back(input_names_.back().c_str());

    // Get output name
    auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator_);
    output_names_.push_back(std::string(output_name_ptr.get()));
    output_node_names_.push_back(output_names_.back().c_str());
  }

  void log_model_info() {
    size_t num_input_nodes = session_->GetInputCount();
    size_t num_output_nodes = session_->GetOutputCount();

    RCLCPP_INFO(this->get_logger(), "=== MODEL INFO ===");
    RCLCPP_INFO(this->get_logger(), "Number of inputs: %zu", num_input_nodes);
    RCLCPP_INFO(this->get_logger(), "Number of outputs: %zu", num_output_nodes);

    // Log input info and extract input size
    auto input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();

    RCLCPP_INFO(this->get_logger(), "=== INPUT TENSOR INFO ===");
    RCLCPP_INFO(this->get_logger(), "Input name: %s",
                input_names_.back().c_str());
    RCLCPP_INFO(this->get_logger(), "Input dimensions: %zu", input_dims.size());

    std::string input_shape = "Input shape: [";
    for (size_t i = 0; i < input_dims.size(); i++) {
      if (i > 0)
        input_shape += ", ";
      input_shape += std::to_string(input_dims[i]);
    }
    input_shape += "]";
    RCLCPP_INFO(this->get_logger(), "%s", input_shape.c_str());

    // Extract input size from model (assuming square input: [batch, channels,
    // height, width])
    if (input_dims.size() >= 4) {
      input_size_ = static_cast<int>(input_dims[2]); // height dimension
      RCLCPP_INFO(this->get_logger(), "Auto-detected input size: %dx%d",
                  input_size_, input_size_);
    } else {
      RCLCPP_ERROR(this->get_logger(),
                   "Unexpected input tensor dimensions. Expected 4D tensor "
                   "[batch, channels, height, width]");
      input_size_ = 640; // fallback
    }

    size_t input_elements = 1;
    for (size_t i = 0; i < input_dims.size(); ++i) {
      input_elements *= input_dims[i];
    }
    RCLCPP_INFO(this->get_logger(), "Input tensor elements: %zu",
                input_elements);

    // Log output info
    auto output_type_info = session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();

    RCLCPP_INFO(this->get_logger(), "=== OUTPUT TENSOR INFO (STATIC) ===");
    RCLCPP_INFO(this->get_logger(), "Output name: %s",
                output_names_.back().c_str());
    RCLCPP_INFO(this->get_logger(), "Output dimensions: %zu",
                output_dims.size());

    std::string output_shape = "Output shape: [";
    for (size_t i = 0; i < output_dims.size(); i++) {
      if (i > 0)
        output_shape += ", ";
      output_shape += std::to_string(output_dims[i]);
    }
    output_shape += "]";
    RCLCPP_INFO(this->get_logger(), "%s", output_shape.c_str());

    size_t output_elements = 1;
    for (size_t i = 0; i < output_dims.size(); ++i) {
      output_elements *= output_dims[i];
    }
    RCLCPP_INFO(this->get_logger(), "Output tensor elements: %zu",
                output_elements);
    RCLCPP_INFO(this->get_logger(), "==================");
  }

  void draw_detection_results(cv::Mat &image,
                              const std::vector<cv::Rect> &boxes,
                              const std::vector<float> &confidences,
                              const std::vector<int> &class_ids,
                              const std::vector<int> &indices) {
    for (int idx : indices) {
      cv::Rect box = boxes[idx];
      float confidence = confidences[idx];
      int class_id = class_ids[idx];

      draw_single_detection(image, box, confidence, class_id);
    }
  }

  void draw_single_detection(cv::Mat &image, const cv::Rect &box,
                             float confidence, int class_id) {
    // Draw rectangle
    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

    // Draw label with class name
    std::string class_name = getClassName(class_id);
    std::string label = class_name + ": " +
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

  void extract_detections(float *output_data, const std::vector<int64_t> &shape,
                          float confidence_threshold, int img_width,
                          int img_height, std::vector<cv::Rect> &boxes,
                          std::vector<float> &confidences,
                          std::vector<int> &class_ids,
                          bool apply_class_filter = true) {
    // YOLOv8 output shape: [1, 84, num_detections] (e.g., 2100 for 320x320,
    // 8400 for 640x640)
    int num_detections = shape[2];
    int num_features = shape[1]; // 84

    for (int i = 0; i < num_detections; ++i) {
      // Access data using correct memory layout: channel * num_detections +
      // anchor
      float cx = output_data[0 * num_detections + i];
      float cy = output_data[1 * num_detections + i];
      float w = output_data[2 * num_detections + i];
      float h = output_data[3 * num_detections + i];

      auto class_result =
          find_best_class(output_data, i, num_features, num_detections);

      if (class_result.confidence > confidence_threshold) {
        // Filter by target classes (only if apply_class_filter is true)
        if (!apply_class_filter || target_classes_.empty() ||
            target_classes_.count(class_result.class_id) > 0) {
          // Convert to actual image coordinates
          int x = static_cast<int>((cx - w / 2) * img_width / input_size_);
          int y = static_cast<int>((cy - h / 2) * img_height / input_size_);
          int width = static_cast<int>(w * img_width / input_size_);
          int height = static_cast<int>(h * img_height / input_size_);

          boxes.emplace_back(x, y, width, height);
          confidences.push_back(class_result.confidence);
          class_ids.push_back(class_result.class_id);
        }
      }
    }
  }

  struct ClassResult {
    float confidence;
    int class_id;
  };

  struct DetectionResults {
    std::map<int, int> class_counts; // class_id -> count
    int total_detections = 0;
  };

  ClassResult find_best_class(float *output_data, int detection_idx,
                              int num_features, int num_detections) {
    float max_class_score = 0.0f;
    int class_id = -1;

    for (int j = 4; j < num_features; ++j) {
      float class_score = output_data[j * num_detections + detection_idx];
      if (class_score > max_class_score) {
        max_class_score = class_score;
        class_id = j - 4;
      }
    }

    return {max_class_score, class_id};
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      // Store input timestamp and frame_id for detection results
      input_timestamp_ = msg->header.stamp;
      input_frame_id_ = msg->header.frame_id;

      // Convert ROS image to OpenCV format
      cv_bridge::CvImagePtr cv_ptr =
          cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat img = cv_ptr->image;

      if (!session_) {
        RCLCPP_WARN(this->get_logger(), "Model not loaded, skipping detection");
        return;
      }

      // Perform object detection and publish result if there are subscribers
      if (image_publisher_->get_subscription_count() > 0) {
        cv::Mat result_img = perform_detection(img, true);
        publish_result_image(result_img);
      } else {
        // Perform detection without drawing (for logging purposes)
        perform_detection(img, false);
      }

    } catch (cv_bridge::Exception &e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  cv::Mat perform_detection(const cv::Mat &image, bool draw_results = true) {
    cv::Mat result;
    if (draw_results) {
      result = image.clone();
    }

    // Timing measurements
    auto total_start = std::chrono::steady_clock::now();
    auto preprocess_start = std::chrono::steady_clock::now();

    try {
      // Preprocess image
      cv::Mat blob;
      cv::dnn::blobFromImage(image, blob, 1.0 / 255.0,
                             cv::Size(input_size_, input_size_),
                             cv::Scalar(0, 0, 0), true, false);

      // Prepare input tensor
      std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
      size_t input_tensor_size = 1 * 3 * input_size_ * input_size_;

      auto input_tensor = Ort::Value::CreateTensor<float>(
          *memory_info_, (float *)blob.data, input_tensor_size,
          input_shape.data(), input_shape.size());

      auto preprocess_end = std::chrono::steady_clock::now();
      auto inference_start = std::chrono::steady_clock::now();

      // Run inference
      auto output_tensors =
          session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(),
                        &input_tensor, 1, output_node_names_.data(), 1);

      auto inference_end = std::chrono::steady_clock::now();
      auto postprocess_start = std::chrono::steady_clock::now();

      // Process output and collect detection results
      DetectionResults detection_results;
      if (!output_tensors.empty()) {
        auto &output_tensor = output_tensors[0];
        auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        float *output_data = output_tensor.GetTensorMutableData<float>();
        if (draw_results) {
          detection_results = process_yolo_output_with_results(
              result, output_data, shape, image.cols, image.rows);
        } else {
          detection_results = process_yolo_output_no_draw(
              output_data, shape, image.cols, image.rows);
        }

        // Publish detection results
        publish_detections(output_data, shape, image.cols, image.rows,
                           input_timestamp_, input_frame_id_);
      }

      auto postprocess_end = std::chrono::steady_clock::now();
      auto total_end = std::chrono::steady_clock::now();

      // Calculate timing
      auto preprocess_ms = std::chrono::duration<double, std::milli>(
                               preprocess_end - preprocess_start)
                               .count();
      auto inference_ms = std::chrono::duration<double, std::milli>(
                              inference_end - inference_start)
                              .count();
      auto postprocess_ms = std::chrono::duration<double, std::milli>(
                                postprocess_end - postprocess_start)
                                .count();
      auto total_ms =
          std::chrono::duration<double, std::milli>(total_end - total_start)
              .count();

      // Log results in PyTorch YOLOv8 format
      log_detection_results(detection_results, image.cols, image.rows, total_ms,
                            preprocess_ms, inference_ms, postprocess_ms);

    } catch (const Ort::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "ONNX Runtime inference error: %s",
                   e.what());
    }

    return result;
  }

  DetectionResults
  process_yolo_output_no_draw(float *output_data,
                              const std::vector<int64_t> &shape, int img_width,
                              int img_height) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    extract_detections(output_data, shape, confidence_threshold_, img_width,
                       img_height, boxes, confidences, class_ids,
                       true); // apply_class_filter = true

    // Apply non-maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_,
                      indices);

    // Collect detection results without drawing
    DetectionResults results;
    results.total_detections = indices.size();

    for (int idx : indices) {
      int class_id = class_ids[idx];
      results.class_counts[class_id]++;
    }

    return results;
  }

  void publish_result_image(const cv::Mat &image) {
    try {
      // Compress image to JPEG
      std::vector<uchar> buffer;
      std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
      cv::imencode(".jpg", image, buffer, params);

      // Create compressed image message
      auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
      msg->header.stamp = this->get_clock()->now();
      msg->header.frame_id = "camera_frame";
      msg->format = "jpeg";
      msg->data = buffer;

      // Publish the compressed image
      image_publisher_->publish(std::move(msg));

    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "Failed to publish result image: %s",
                  e.what());
    }
  }

  void publish_detections(float *output_data, const std::vector<int64_t> &shape,
                          int img_width, int img_height,
                          const rclcpp::Time &timestamp,
                          const std::string &frame_id) {
    try {
      std::vector<cv::Rect> boxes;
      std::vector<float> confidences;
      std::vector<int> class_ids;

      extract_detections(output_data, shape, confidence_threshold_, img_width,
                         img_height, boxes, confidences, class_ids,
                         true); // apply_class_filter = true

      // Apply non-maximum suppression
      std::vector<int> indices;
      cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_,
                        nms_threshold_, indices);

      // Create Detection2DArray message
      auto detection_msg =
          std::make_unique<vision_msgs::msg::Detection2DArray>();
      detection_msg->header.stamp = timestamp;
      detection_msg->header.frame_id =
          frame_id.empty() ? "camera_frame" : frame_id;

      // Add detections (even if empty)
      for (int idx : indices) {
        vision_msgs::msg::Detection2D detection;

        // Set bounding box
        detection.bbox.center.position.x =
            boxes[idx].x + boxes[idx].width / 2.0;
        detection.bbox.center.position.y =
            boxes[idx].y + boxes[idx].height / 2.0;
        detection.bbox.center.theta = 0.0;
        detection.bbox.size_x = boxes[idx].width;
        detection.bbox.size_y = boxes[idx].height;

        // Set detection hypothesis
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        hypothesis.hypothesis.class_id = std::to_string(class_ids[idx]);
        hypothesis.hypothesis.score = confidences[idx];
        detection.results.push_back(hypothesis);

        detection_msg->detections.push_back(detection);
      }

      // Publish detection results (always publish, even if empty)
      detection_publisher_->publish(std::move(detection_msg));

    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "Failed to publish detections: %s",
                  e.what());
    }
  }

  void process_yolo_output(cv::Mat &image, float *output_data,
                           const std::vector<int64_t> &shape, int img_width,
                           int img_height) {

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    extract_detections(output_data, shape, confidence_threshold_, img_width,
                       img_height, boxes, confidences, class_ids,
                       false); // apply_class_filter = false

    // Apply non-maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_,
                      indices);

    draw_detection_results(image, boxes, confidences, class_ids, indices);
  }

  DetectionResults
  process_yolo_output_with_results(cv::Mat &image, float *output_data,
                                   const std::vector<int64_t> &shape,
                                   int img_width, int img_height) {
    // For drawing: get all objects (no class filtering)
    std::vector<cv::Rect> all_boxes;
    std::vector<float> all_confidences;
    std::vector<int> all_class_ids;

    extract_detections(output_data, shape, confidence_threshold_, img_width,
                       img_height, all_boxes, all_confidences, all_class_ids,
                       false); // apply_class_filter = false

    // Apply non-maximum suppression to all objects
    std::vector<int> all_indices;
    cv::dnn::NMSBoxes(all_boxes, all_confidences, confidence_threshold_,
                      nms_threshold_, all_indices);

    draw_detection_results(image, all_boxes, all_confidences, all_class_ids,
                           all_indices);

    // For results: get filtered objects only
    std::vector<cv::Rect> filtered_boxes;
    std::vector<float> filtered_confidences;
    std::vector<int> filtered_class_ids;

    extract_detections(output_data, shape, confidence_threshold_, img_width,
                       img_height, filtered_boxes, filtered_confidences,
                       filtered_class_ids, true); // apply_class_filter = true

    // Apply non-maximum suppression to filtered objects
    std::vector<int> filtered_indices;
    cv::dnn::NMSBoxes(filtered_boxes, filtered_confidences,
                      confidence_threshold_, nms_threshold_, filtered_indices);

    // Collect detection results from filtered objects
    DetectionResults results;
    results.total_detections = filtered_indices.size();

    for (int idx : filtered_indices) {
      int class_id = filtered_class_ids[idx];
      results.class_counts[class_id]++;
    }

    return results;
  }

  void log_detection_results(const DetectionResults &results, int img_width,
                             int img_height, double total_ms,
                             double preprocess_ms, double inference_ms,
                             double postprocess_ms) {
    // Build detection summary string with class names (e.g., "2 boats, 1
    // handbag, 1 couch, 1 bed")
    std::stringstream detection_summary;

    if (results.total_detections == 0) {
      detection_summary << "(no detections)";
    } else {
      bool first = true;
      for (const auto &pair : results.class_counts) {
        if (!first)
          detection_summary << ", ";
        std::string class_name = getClassName(pair.first);
        std::string plural_suffix = (pair.second > 1) ? "s" : "";
        detection_summary << pair.second << " " << class_name << plural_suffix;
        first = false;
      }
    }

    // Format: "0: 480x640 2 boats, 1 handbag, 1 couch, 1 bed, 134.8ms"
    RCLCPP_INFO(this->get_logger(), "0: %dx%d %s, %.1fms", img_width,
                img_height, detection_summary.str().c_str(), total_ms);

    // Format: "Speed: 1.0ms preprocess, 134.8ms inference, 0.7ms postprocess
    // per image at shape (1, 3, 320, 320)"
    RCLCPP_INFO(this->get_logger(),
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms "
                "postprocess per image at shape (1, 3, %d, %d)",
                preprocess_ms, inference_ms, postprocess_ms, input_size_,
                input_size_);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr
      image_publisher_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr
      detection_publisher_;

  // Model parameters
  std::string model_path_;
  int input_size_; // Auto-detected from model

  // Inference parameters
  double confidence_threshold_;
  double nms_threshold_;
  std::set<int> target_classes_;

  // Runtime parameters
  int num_threads_;

  // I/O parameters
  std::string input_topic_;
  std::string output_topic_;

  // ONNX Runtime components
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char *> input_node_names_;
  std::vector<const char *> output_node_names_;

  // COCO class labels
  std::vector<std::string> coco_labels_;

  // Input timestamp and frame_id for detection results
  rclcpp::Time input_timestamp_;
  std::string input_frame_id_;
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
