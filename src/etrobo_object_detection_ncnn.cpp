#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <iomanip>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sstream>
#include <vector>

#include <ncnn/layer.h>
#include <ncnn/net.h>

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
};

class ObjectDetectionNCNNNode : public rclcpp::Node {
public:
  ObjectDetectionNCNNNode() : Node("object_detection_ncnn") {
    initializeCocoLabels();
    declare_parameters();

    try {
      initialize_ncnn_network();
      log_model_info();
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error loading model: %s", e.what());
      return;
    }

    setup_subscription();
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
    this->declare_parameter("param_path", "yolov8n.ncnn.param");
    this->declare_parameter("bin_path", "yolov8n.ncnn.bin");
    this->declare_parameter("confidence_threshold", 0.25);
    this->declare_parameter("nms_threshold", 0.4);
    this->declare_parameter("num_threads", 1);
    this->declare_parameter("input_topic", "/image_raw");
    this->declare_parameter("display_results", true);
    this->declare_parameter("use_vulkan", true);

    param_path_ = this->get_parameter("param_path").as_string();
    bin_path_ = this->get_parameter("bin_path").as_string();
    confidence_threshold_ =
        this->get_parameter("confidence_threshold").as_double();
    nms_threshold_ = this->get_parameter("nms_threshold").as_double();
    num_threads_ = this->get_parameter("num_threads").as_int();
    input_topic_ = this->get_parameter("input_topic").as_string();
    display_results_ = this->get_parameter("display_results").as_bool();
    use_vulkan_ = this->get_parameter("use_vulkan").as_bool();

    RCLCPP_INFO(this->get_logger(), "Parameters:");
    RCLCPP_INFO(this->get_logger(), "  param_path: %s", param_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "  bin_path: %s", bin_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "  confidence_threshold: %.2f",
                confidence_threshold_);
    RCLCPP_INFO(this->get_logger(), "  nms_threshold: %.2f", nms_threshold_);
    RCLCPP_INFO(this->get_logger(), "  num_threads: %d", num_threads_);
    RCLCPP_INFO(this->get_logger(), "  input_topic: %s", input_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  display_results: %s",
                display_results_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  use_vulkan: %s",
                use_vulkan_ ? "true" : "false");
  }

  void setup_subscription() {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.best_effort();

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        input_topic_, qos,
        std::bind(&ObjectDetectionNCNNNode::image_callback, this,
                  std::placeholders::_1));
  }

  void initialize_ncnn_network() {
    net_ = std::make_unique<ncnn::Net>();

    net_->opt.use_vulkan_compute = use_vulkan_;
    net_->opt.num_threads = num_threads_;

    int ret = net_->load_param(param_path_.c_str());
    if (ret != 0) {
      throw std::runtime_error("Failed to load param file: " + param_path_);
    }

    ret = net_->load_model(bin_path_.c_str());
    if (ret != 0) {
      throw std::runtime_error("Failed to load model file: " + bin_path_);
    }

    RCLCPP_INFO(this->get_logger(), "NCNN YOLO model loaded successfully");
  }

  void log_model_info() {
    RCLCPP_INFO(this->get_logger(), "=== MODEL INFO ===");
    RCLCPP_INFO(this->get_logger(), "Param file: %s", param_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Model file: %s", bin_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Vulkan enabled: %s",
                use_vulkan_ ? "true" : "false");
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
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
      while (objects[i].prob > p)
        i++;

      while (objects[j].prob < p)
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

      int keep = 1;
      for (int j = 0; j < (int)picked.size(); j++) {
        const Object &b = objects[picked[j]];

        float inter_area = intersection_area(a, b);
        float union_area = areas[i] + areas[picked[j]] - inter_area;
        if (inter_area / union_area > nms_threshold)
          keep = 0;
      }

      if (keep)
        picked.push_back(i);
    }
  }

  static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

  void generate_proposals(const ncnn::Mat &pred, int stride,
                          const ncnn::Mat &in_pad, float prob_threshold,
                          std::vector<Object> &objects) {
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4;

    for (int y = 0; y < num_grid_y; y++) {
      for (int x = 0; x < num_grid_x; x++) {
        const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

        int label = -1;
        float score = -FLT_MAX;
        {
          const ncnn::Mat pred_score =
              pred_grid.range(reg_max_1 * 4, num_class);

          for (int k = 0; k < num_class; k++) {
            float s = pred_score[k];
            if (s > score) {
              label = k;
              score = s;
            }
          }

          score = sigmoid(score);
        }

        if (score >= prob_threshold) {
          ncnn::Mat pred_bbox =
              pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4);

          {
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
          }

          float pred_ltrb[4];
          for (int k = 0; k < 4; k++) {
            float dis = 0.f;
            const float *dis_after_sm = pred_bbox.row(k);
            for (int l = 0; l < reg_max_1; l++) {
              dis += l * dis_after_sm[l];
            }
            pred_ltrb[k] = dis * stride;
          }

          float pb_cx = (x + 0.5f) * stride;
          float pb_cy = (y + 0.5f) * stride;

          float x0 = pb_cx - pred_ltrb[0];
          float y0 = pb_cy - pred_ltrb[1];
          float x1 = pb_cx + pred_ltrb[2];
          float y1 = pb_cy + pred_ltrb[3];

          Object obj;
          obj.rect.x = x0;
          obj.rect.y = y0;
          obj.rect.width = x1 - x0;
          obj.rect.height = y1 - y0;
          obj.label = label;
          obj.prob = score;

          objects.push_back(obj);
        }
      }
    }
  }

  void generate_proposals(const ncnn::Mat &pred,
                          const std::vector<int> &strides,
                          const ncnn::Mat &in_pad, float prob_threshold,
                          std::vector<Object> &objects) {
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++) {
      const int stride = strides[i];

      const int num_grid_x = w / stride;
      const int num_grid_y = h / stride;
      const int num_grid = num_grid_x * num_grid_y;

      generate_proposals(pred.row_range(pred_row_offset, num_grid), stride,
                         in_pad, prob_threshold, objects);
      pred_row_offset += num_grid;
    }
  }

  int detect_yolov8(const cv::Mat &bgr, std::vector<Object> &objects) {
    const int target_size = 320;
    const float prob_threshold = confidence_threshold_;
    const float nms_threshold = nms_threshold_;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
      scale = (float)target_size / w;
      w = target_size;
      h = h * scale;
    } else {
      scale = (float)target_size / h;
      h = target_size;
      w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,
                           wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net_->create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Object> proposals;
    generate_proposals(out, strides, in_pad, prob_threshold, proposals);

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++) {
      objects[i] = proposals[picked[i]];

      float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
      float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
      float x1 =
          (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
      float y1 =
          (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

      x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
      y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
      x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
      y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

      objects[i].rect.x = x0;
      objects[i].rect.y = y0;
      objects[i].rect.width = x1 - x0;
      objects[i].rect.height = y1 - y0;
    }

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
                "postprocess per image at shape (1, 3, 320, 320)",
                preprocess_ms, inference_ms, postprocess_ms);
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      cv_bridge::CvImagePtr cv_ptr =
          cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat img = cv_ptr->image;

      if (!net_) {
        RCLCPP_WARN(this->get_logger(), "Model not loaded, skipping detection");
        return;
      }

      cv::Mat result_img = perform_detection(img);

      if (display_results_) {
        cv::imshow("result", result_img);
        cv::waitKey(1);
      }

    } catch (cv_bridge::Exception &e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  cv::Mat perform_detection(const cv::Mat &image) {
    cv::Mat result = image.clone();

    auto total_start = std::chrono::steady_clock::now();
    auto preprocess_start = std::chrono::steady_clock::now();

    std::vector<Object> objects;
    auto preprocess_end = std::chrono::steady_clock::now();
    auto inference_start = std::chrono::steady_clock::now();

    detect_yolov8(image, objects);

    auto inference_end = std::chrono::steady_clock::now();
    auto postprocess_start = std::chrono::steady_clock::now();

    draw_detection_results(result, objects);
    DetectionResults detection_results = process_detection_results(objects);

    auto postprocess_end = std::chrono::steady_clock::now();
    auto total_end = std::chrono::steady_clock::now();

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

    log_detection_results(detection_results, image.cols, image.rows, total_ms,
                          preprocess_ms, inference_ms, postprocess_ms);

    return result;
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;

  std::string param_path_;
  std::string bin_path_;
  double confidence_threshold_;
  double nms_threshold_;
  int num_threads_;
  std::string input_topic_;
  bool display_results_;
  bool use_vulkan_;

  std::unique_ptr<ncnn::Net> net_;
  std::vector<std::string> coco_labels_;
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
