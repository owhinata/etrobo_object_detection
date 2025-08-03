#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <iomanip>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <set>
#include <sstream>
#include <std_msgs/msg/header.hpp>
#include <vector>
#include <vision_msgs/msg/bounding_box2_d.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include <ncnn/layer.h>
#include <ncnn/net.h>

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
};

struct PreprocessResult {
  ncnn::Mat input_tensor;
  float scale;
  int wpad;
  int hpad;
  int original_width;
  int original_height;
};

class ObjectDetectionNCNNNode : public rclcpp::Node {
private:
  // YOLO model constants
  static constexpr int REG_MAX = 16;
  static constexpr int STRIDE_8 = 8;
  static constexpr int STRIDE_16 = 16;
  static constexpr int STRIDE_32 = 32;
  static constexpr int MAX_STRIDE = 32;
  static constexpr float PADDING_VALUE = 114.0f;
  static constexpr float NORMALIZATION_FACTOR = 1.0f / 255.0f;

  // Helper function for timing calculations
  double
  calculate_duration_ms(const std::chrono::steady_clock::time_point &start,
                        const std::chrono::steady_clock::time_point &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  // Helper function to clamp coordinate to image bounds
  float clamp_to_image_bounds(float value, float max_value) {
    return std::max(std::min(value, max_value - 1.0f), 0.0f);
  }

  // Filter objects by target classes
  std::vector<Object>
  filter_objects_by_target_classes(const std::vector<Object> &all_objects) {
    std::vector<Object> filtered_objects;
    for (const auto &obj : all_objects) {
      if (target_classes_.empty() || target_classes_.count(obj.label) > 0) {
        filtered_objects.push_back(obj);
      }
    }
    return filtered_objects;
  }

  // Draw results on image if requested
  cv::Mat draw_results_if_needed(const cv::Mat &image,
                                 const std::vector<Object> &all_objects,
                                 bool draw_results) {
    cv::Mat result;
    if (draw_results) {
      result = image.clone();
      draw_detection_results(result, all_objects);
    }
    return result;
  }

  // Preprocess image for YOLO inference
  PreprocessResult preprocess_image(const cv::Mat &bgr) {
    PreprocessResult result;
    result.original_width = bgr.cols;
    result.original_height = bgr.rows;

    const int target_size = input_size_;
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // Calculate target dimensions
    int w = img_w;
    int h = img_h;
    if (w > h) {
      result.scale = (float)target_size / w;
      w = target_size;
      h = h * result.scale;
    } else {
      result.scale = (float)target_size / h;
      h = target_size;
      w = w * result.scale;
    }

    // Calculate padding
    result.wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    result.hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    int final_w = w + result.wpad;
    int final_h = h + result.hpad;

    // Apply center padding if needed (letterbox effect)
    if (result.wpad > 0 || result.hpad > 0) {
      // Create a resized image first, then add padding
      ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
          bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
      ncnn::copy_make_border(resized, result.input_tensor, result.hpad / 2,
                             result.hpad - result.hpad / 2, result.wpad / 2,
                             result.wpad - result.wpad / 2,
                             ncnn::BORDER_CONSTANT, PADDING_VALUE);
    } else {
      result.input_tensor = ncnn::Mat::from_pixels_resize(
          bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, final_w, final_h);
    }

    // Normalization
    const float norm_vals[3] = {NORMALIZATION_FACTOR, NORMALIZATION_FACTOR,
                                NORMALIZATION_FACTOR};
    result.input_tensor.substract_mean_normalize(0, norm_vals);

    return result;
  }

  // Run NCNN inference
  ncnn::Mat run_inference(const ncnn::Mat &input_tensor) {
    ncnn::Extractor ex = net_->create_extractor();
    ex.input("in0", input_tensor);

    ncnn::Mat output;
    ex.extract("out0", output);
    return output;
  }

  // Apply NMS and coordinate transformation
  void apply_nms_and_transform(std::vector<Object> &proposals,
                               const PreprocessResult &preprocess_result,
                               std::vector<Object> &objects) {
    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold_);

    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++) {
      objects[i] = proposals[picked[i]];

      // Transform coordinates back to original image space
      float x0 = (objects[i].rect.x - (preprocess_result.wpad / 2)) /
                 preprocess_result.scale;
      float y0 = (objects[i].rect.y - (preprocess_result.hpad / 2)) /
                 preprocess_result.scale;
      float x1 = (objects[i].rect.x + objects[i].rect.width -
                  (preprocess_result.wpad / 2)) /
                 preprocess_result.scale;
      float y1 = (objects[i].rect.y + objects[i].rect.height -
                  (preprocess_result.hpad / 2)) /
                 preprocess_result.scale;

      x0 = clamp_to_image_bounds(
          x0, static_cast<float>(preprocess_result.original_width));
      y0 = clamp_to_image_bounds(
          y0, static_cast<float>(preprocess_result.original_height));
      x1 = clamp_to_image_bounds(
          x1, static_cast<float>(preprocess_result.original_width));
      y1 = clamp_to_image_bounds(
          y1, static_cast<float>(preprocess_result.original_height));

      objects[i].rect.x = x0;
      objects[i].rect.y = y0;
      objects[i].rect.width = x1 - x0;
      objects[i].rect.height = y1 - y0;
    }
  }

public:
  ObjectDetectionNCNNNode() : Node("object_detection_ncnn") {
    initializeCocoLabels();
    declare_parameters();
    get_parameters();
    log_parameters();

    try {
      initialize_ncnn_network();
      log_model_info();
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
    this->declare_parameter("model_path", "yolov8n.ncnn.bin");
    this->declare_parameter("input_size", 320);

    // Inference parameters
    this->declare_parameter("confidence_threshold", 0.25);
    this->declare_parameter("nms_threshold", 0.4);
    this->declare_parameter("target_classes",
                            std::vector<int64_t>{}); // all objects

    // Runtime parameters
    this->declare_parameter("num_threads", 1);

    // I/O parameters
    this->declare_parameter("input_topic", "/image_raw");
    this->declare_parameter("output_topic", "/object_detection");
  }

  void get_parameters() {
    // Get model parameters
    model_path_ = this->get_parameter("model_path").as_string();
    input_size_ = this->get_parameter("input_size").as_int();

    // Generate param path by changing extension from .bin to .param
    generate_param_path();

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
  }

  void generate_param_path() {
    param_path_ = model_path_;
    size_t last_dot = param_path_.find_last_of(".");
    if (last_dot != std::string::npos) {
      param_path_ = param_path_.substr(0, last_dot) + ".param";
    } else {
      param_path_ = param_path_ + ".param";
    }
  }

  void log_parameters() {
    RCLCPP_INFO(this->get_logger(), "Parameters:");
    // Model parameters
    RCLCPP_INFO(this->get_logger(), "  model_path: %s", model_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "  input_size: %d", input_size_);

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
    log_target_classes();
  }

  void log_target_classes() {
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

  rclcpp::QoS create_default_qos() {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.best_effort();
    return qos;
  }

  rclcpp::QoS create_reliable_qos() {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.reliable();
    return qos;
  }

  void setup_subscription() {
    auto qos = create_default_qos();

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        input_topic_, qos,
        std::bind(&ObjectDetectionNCNNNode::image_callback, this,
                  std::placeholders::_1));
  }

  void setup_publisher() {
    auto default_qos = create_default_qos();
    auto reliable_qos = create_reliable_qos();

    // Raw image publisher only with reliable QoS
    image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        output_topic_ + "/image", reliable_qos);

    detection_publisher_ =
        this->create_publisher<vision_msgs::msg::Detection2DArray>(
            output_topic_ + "/detections", default_qos);
  }

  void initialize_ncnn_network() {
    net_ = std::make_unique<ncnn::Net>();

    net_->opt.use_vulkan_compute = true;
    net_->opt.num_threads = num_threads_;

    int ret = net_->load_param(param_path_.c_str());
    if (ret != 0) {
      throw std::runtime_error("Failed to load param file: " + param_path_);
    }

    ret = net_->load_model(model_path_.c_str());
    if (ret != 0) {
      throw std::runtime_error("Failed to load model file: " + model_path_);
    }

    RCLCPP_INFO(this->get_logger(), "NCNN YOLO model loaded successfully");
  }

  void log_model_info() {
    RCLCPP_INFO(this->get_logger(), "=== MODEL INFO ===");
    RCLCPP_INFO(this->get_logger(), "Param file: %s", param_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Model file: %s", model_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Threads: %d", num_threads_);
    RCLCPP_INFO(this->get_logger(), "==================");
  }

  static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
  }

  static void qsort_descent_inplace(std::vector<Object> &objects, int left,
                                    int right) {
    int i = left;
    int j = right;
    float pivot = objects[(left + right) / 2].prob;

    while (i <= j) {
      while (objects[i].prob > pivot)
        i++;

      while (objects[j].prob < pivot)
        j--;

      if (i <= j) {
        std::swap(objects[i], objects[j]);
        i++;
        j--;
      }
    }

    if (left < j)
      qsort_descent_inplace(objects, left, j);
    if (i < right)
      qsort_descent_inplace(objects, i, right);
  }

  static void qsort_descent_inplace(std::vector<Object> &objects) {
    if (objects.empty())
      return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
  }

  static void nms_sorted_bboxes(const std::vector<Object> &objects,
                                std::vector<int> &picked, float nms_threshold) {
    picked.clear();

    const int n = objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
      areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
      const Object &a = objects[i];

      bool should_keep = true;
      for (int j = 0; j < (int)picked.size(); j++) {
        const Object &b = objects[picked[j]];

        float inter_area = intersection_area(a, b);
        float union_area = areas[i] + areas[picked[j]] - inter_area;
        if (inter_area / union_area > nms_threshold)
          should_keep = false;
      }

      if (should_keep)
        picked.push_back(i);
    }
  }

  static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

  // Calculate class score and label for a grid prediction
  std::pair<int, float> calculate_class_score(const ncnn::Mat &pred_grid,
                                              int num_class) {
    const ncnn::Mat pred_score = pred_grid.range(REG_MAX * 4, num_class);

    int label = -1;
    float score = -FLT_MAX;

    for (int k = 0; k < num_class; k++) {
      float class_score = pred_score[k];
      if (class_score > score) {
        label = k;
        score = class_score;
      }
    }

    return std::make_pair(label, sigmoid(score));
  }

  // Calculate bounding box coordinates from prediction
  void calculate_bbox_coordinates(const ncnn::Mat &pred_grid, int stride,
                                  int grid_x, int grid_y,
                                  float bbox_coords[4]) {
    ncnn::Mat pred_bbox = pred_grid.range(0, REG_MAX * 4).reshape(REG_MAX, 4);

    // Apply softmax to regression values
    ncnn::Layer *softmax = ncnn::create_layer("Softmax");
    ncnn::ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 1);
    softmax->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    softmax->create_pipeline(opt);
    softmax->forward_inplace(pred_bbox, opt);
    softmax->destroy_pipeline(opt);
    delete softmax;

    // Calculate distance values
    float pred_ltrb[4];
    for (int k = 0; k < 4; k++) {
      float dis = 0.f;
      const float *dis_after_sm = pred_bbox.row(k);
      for (int l = 0; l < REG_MAX; l++) {
        dis += l * dis_after_sm[l];
      }
      pred_ltrb[k] = dis * stride;
    }

    // Calculate center point and final coordinates
    float pb_cx = (grid_x + 0.5f) * stride;
    float pb_cy = (grid_y + 0.5f) * stride;

    bbox_coords[0] = pb_cx - pred_ltrb[0]; // x0
    bbox_coords[1] = pb_cy - pred_ltrb[1]; // y0
    bbox_coords[2] = pb_cx + pred_ltrb[2]; // x1
    bbox_coords[3] = pb_cy + pred_ltrb[3]; // y1
  }

  void generate_proposals(const ncnn::Mat &pred, int stride,
                          const ncnn::Mat &in_pad, float prob_threshold,
                          std::vector<Object> &objects,
                          bool apply_class_filter = true) {
    const int w = in_pad.w;
    const int h = in_pad.h;
    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;
    const int num_class = pred.w - REG_MAX * 4;

    for (int y = 0; y < num_grid_y; y++) {
      for (int x = 0; x < num_grid_x; x++) {
        const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

        // Calculate class score and label
        auto class_result = calculate_class_score(pred_grid, num_class);
        int label = class_result.first;
        float score = class_result.second;

        // Skip if score is below threshold
        if (score < prob_threshold) {
          continue;
        }

        // Skip if class filtering is enabled and class is not in target list
        if (apply_class_filter && !target_classes_.empty() &&
            target_classes_.count(label) == 0) {
          continue;
        }

        // Calculate bounding box coordinates
        float bbox_coords[4];
        calculate_bbox_coordinates(pred_grid, stride, x, y, bbox_coords);

        // Create object and add to results
        Object obj;
        obj.rect.x = bbox_coords[0];
        obj.rect.y = bbox_coords[1];
        obj.rect.width = bbox_coords[2] - bbox_coords[0];
        obj.rect.height = bbox_coords[3] - bbox_coords[1];
        obj.label = label;
        obj.prob = score;

        objects.push_back(obj);
      }
    }
  }

  void generate_proposals(const ncnn::Mat &pred,
                          const std::vector<int> &strides,
                          const ncnn::Mat &in_pad, float prob_threshold,
                          std::vector<Object> &objects,
                          bool apply_class_filter = true) {
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++) {
      const int stride = strides[i];

      const int num_grid_x = w / stride;
      const int num_grid_y = h / stride;
      const int num_grid = num_grid_x * num_grid_y;

      generate_proposals(pred.row_range(pred_row_offset, num_grid), stride,
                         in_pad, prob_threshold, objects, apply_class_filter);
      pred_row_offset += num_grid;
    }
  }

  int detect_yolov8(const cv::Mat &bgr, std::vector<Object> &objects,
                    bool apply_class_filter = true) {
    // Step 1: Preprocess image
    PreprocessResult preprocess_result = preprocess_image(bgr);

    // Step 2: Run inference
    ncnn::Mat output = run_inference(preprocess_result.input_tensor);

    // Step 3: Generate proposals
    std::vector<int> strides = {STRIDE_8, STRIDE_16, STRIDE_32};
    std::vector<Object> proposals;
    generate_proposals(output, strides, preprocess_result.input_tensor,
                       confidence_threshold_, proposals, apply_class_filter);

    // Step 4: Apply NMS and coordinate transformation
    apply_nms_and_transform(proposals, preprocess_result, objects);

    return 0;
  }

  void draw_detection_results(cv::Mat &image,
                              const std::vector<Object> &objects) {
    for (const auto &obj : objects) {
      cv::rectangle(
          image,
          cv::Rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height),
          cv::Scalar(0, 255, 0), 2);

      std::string class_name = getClassName(obj.label);
      std::string label = class_name + ": " +
                          std::to_string(static_cast<int>(obj.prob * 100)) +
                          "%";

      int baseline;
      cv::Size label_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

      cv::rectangle(image,
                    cv::Point(obj.rect.x, obj.rect.y - label_size.height - 10),
                    cv::Point(obj.rect.x + label_size.width, obj.rect.y),
                    cv::Scalar(0, 255, 0), -1);

      cv::putText(image, label, cv::Point(obj.rect.x, obj.rect.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
  }

  struct DetectionResults {
    std::map<int, int> class_counts;
    int total_detections = 0;
  };

  DetectionResults
  process_detection_results(const std::vector<Object> &objects) {
    DetectionResults results;
    results.total_detections = objects.size();

    for (const auto &obj : objects) {
      results.class_counts[obj.label]++;
    }

    return results;
  }

  void log_detection_results(const DetectionResults &results, int img_width,
                             int img_height, double total_ms,
                             double preprocess_ms, double inference_ms,
                             double postprocess_ms) {
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

    RCLCPP_INFO(this->get_logger(), "0: %dx%d %s, %.1fms", img_width,
                img_height, detection_summary.str().c_str(), total_ms);

    RCLCPP_INFO(this->get_logger(),
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms "
                "postprocess per image at shape (1, 3, %d, %d)",
                preprocess_ms, inference_ms, postprocess_ms, input_size_,
                input_size_);
  }

  // Core detection pipeline without timing measurements
  struct DetectionPipelineResult {
    std::vector<Object> all_objects;
    std::vector<Object> filtered_objects;
    DetectionResults detection_results;
  };

  DetectionPipelineResult run_detection_pipeline(const cv::Mat &image) {
    DetectionPipelineResult result;

    // Run detection without class filtering to get all objects
    detect_yolov8(image, result.all_objects, false);

    // Filter objects by target classes
    result.filtered_objects =
        filter_objects_by_target_classes(result.all_objects);

    // Process detection results for logging
    result.detection_results =
        process_detection_results(result.filtered_objects);

    return result;
  }

  cv::Mat perform_detection(const cv::Mat &image, bool draw_results = true) {
    auto total_start = std::chrono::steady_clock::now();
    auto preprocess_start = std::chrono::steady_clock::now();
    auto preprocess_end = std::chrono::steady_clock::now();

    auto inference_start = std::chrono::steady_clock::now();
    // Run core detection pipeline
    DetectionPipelineResult pipeline_result = run_detection_pipeline(image);
    auto inference_end = std::chrono::steady_clock::now();

    auto postprocess_start = std::chrono::steady_clock::now();
    // Draw results if needed
    cv::Mat result = draw_results_if_needed(image, pipeline_result.all_objects,
                                            draw_results);
    auto postprocess_end = std::chrono::steady_clock::now();

    auto total_end = std::chrono::steady_clock::now();

    // Calculate and log timing
    auto preprocess_ms =
        calculate_duration_ms(preprocess_start, preprocess_end);
    auto inference_ms = calculate_duration_ms(inference_start, inference_end);
    auto postprocess_ms =
        calculate_duration_ms(postprocess_start, postprocess_end);
    auto total_ms = calculate_duration_ms(total_start, total_end);

    log_detection_results(pipeline_result.detection_results, image.cols,
                          image.rows, total_ms, preprocess_ms, inference_ms,
                          postprocess_ms);

    // Publish detection results (filtered objects only)
    publish_detections(pipeline_result.filtered_objects, input_timestamp_,
                       input_frame_id_);

    return result;
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      // Store input metadata for detection results
      input_timestamp_ = msg->header.stamp;
      input_frame_id_ = msg->header.frame_id;

      // Debug: Log input timestamp and frame_id
      RCLCPP_DEBUG(this->get_logger(),
                   "Input timestamp: %.0f.%09ld, frame_id: '%s'",
                   input_timestamp_.seconds(), input_timestamp_.nanoseconds(),
                   input_frame_id_.c_str());

      // Check if model is loaded
      if (!net_) {
        RCLCPP_WARN(this->get_logger(), "Model not loaded, skipping detection");
        return;
      }

      // Convert ROS image to OpenCV format
      cv::Mat image = convert_ros_image_to_cv(msg);

      // Process detection and publish results
      process_detection_with_publishing(image);

    } catch (cv_bridge::Exception &e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  // Convert ROS image message to OpenCV Mat
  cv::Mat
  convert_ros_image_to_cv(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    return cv_ptr->image;
  }

  // Process detection based on subscriber count
  void process_detection_with_publishing(const cv::Mat &image) {
    if (image_publisher_->get_subscription_count() > 0) {
      cv::Mat result_img = perform_detection(image, true);
      publish_result_image(result_img);
    } else {
      // Perform detection without drawing (for logging purposes)
      perform_detection(image, false);
    }
  }

  void publish_result_image(const cv::Mat &image) {
    try {
      // Check if image is valid before publishing
      if (image.empty() || image.cols == 0 || image.rows == 0) {
        RCLCPP_WARN(this->get_logger(),
                    "Attempted to publish empty image, skipping");
        return;
      }

      // Ensure image is in correct format
      cv::Mat publish_image;
      if (image.channels() == 3 && image.type() == CV_8UC3) {
        publish_image = image;
      } else {
        RCLCPP_WARN(this->get_logger(),
                    "Image format conversion needed: channels=%d, type=%d",
                    image.channels(), image.type());
        if (image.channels() == 1) {
          cv::cvtColor(image, publish_image, cv::COLOR_GRAY2BGR);
        } else if (image.channels() == 4) {
          cv::cvtColor(image, publish_image, cv::COLOR_BGRA2BGR);
        } else {
          image.copyTo(publish_image);
        }
      }

      // Publish raw image only
      auto msg =
          cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", publish_image)
              .toImageMsg();
      msg->header.stamp = this->get_clock()->now();
      msg->header.frame_id = "camera_frame";
      image_publisher_->publish(*msg);

    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "Failed to publish result image: %s",
                  e.what());
    }
  }

  void publish_detections(const std::vector<Object> &objects,
                          const rclcpp::Time &timestamp,
                          const std::string &frame_id) {
    try {
      // Create Detection2DArray message
      auto detection_msg =
          std::make_unique<vision_msgs::msg::Detection2DArray>();
      detection_msg->header.stamp = timestamp;
      detection_msg->header.frame_id =
          frame_id.empty() ? "camera_frame" : frame_id;

      // Add detections (even if empty)
      for (const auto &obj : objects) {
        vision_msgs::msg::Detection2D detection;

        // Set bounding box
        detection.bbox.center.position.x = obj.rect.x + obj.rect.width / 2.0;
        detection.bbox.center.position.y = obj.rect.y + obj.rect.height / 2.0;
        detection.bbox.center.theta = 0.0;
        detection.bbox.size_x = obj.rect.width;
        detection.bbox.size_y = obj.rect.height;

        // Set detection hypothesis
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        hypothesis.hypothesis.class_id = std::to_string(obj.label);
        hypothesis.hypothesis.score = obj.prob;
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

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr
      detection_publisher_;

  // Model parameters
  std::string model_path_;
  std::string param_path_; // Auto-generated from model_path
  int input_size_;

  // Inference parameters
  double confidence_threshold_;
  double nms_threshold_;
  std::set<int> target_classes_;

  // Runtime parameters
  int num_threads_;

  // I/O parameters
  std::string input_topic_;
  std::string output_topic_;

  std::unique_ptr<ncnn::Net> net_;
  std::vector<std::string> coco_labels_;

  // Input timestamp and frame_id for detection results
  rclcpp::Time input_timestamp_;
  std::string input_frame_id_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObjectDetectionNCNNNode>();

  try {
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node->get_logger(), "Exception in main: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}
