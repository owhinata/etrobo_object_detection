#include <cv_bridge/cv_bridge.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class ObjectDetectionNode : public rclcpp::Node {
public:
  ObjectDetectionNode() : Node("object_detection") {
    try {
      // Load YOLOv8 model (you'll need to convert the model to ONNX format)
      net_ = cv::dnn::readNetFromONNX("yolov8n.onnx");
      if (net_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load YOLO model");
        return;
      }

      // Set backend and target
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

      RCLCPP_INFO(this->get_logger(), "YOLO model loaded successfully");
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error loading model: %s", e.what());
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

      if (net_.empty()) {
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

    // Prepare input blob
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(640, 640),
                           cv::Scalar(0, 0, 0), true, false);

    // Set input to the network
    net_.setInput(blob);

    // Run inference
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    // Process outputs and draw bounding boxes
    if (!outputs.empty()) {
      process_outputs(result, outputs, image.cols, image.rows);
    }

    return result;
  }

  void process_outputs(cv::Mat &image, const std::vector<cv::Mat> &outputs,
                       int img_width, int img_height) {
    const float confidence_threshold = 0.5;
    const float nms_threshold = 0.4;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // Parse detection outputs
    for (const auto &output : outputs) {
      const int rows = output.size[1];
      const int dimensions = output.size[2];

      for (int i = 0; i < rows; ++i) {
        const float *data = output.ptr<float>(0, i);

        // Extract confidence scores for all classes (skip first 4 elements: x,
        // y, w, h)
        float max_confidence = 0.0f;
        int class_id = -1;
        for (int j = 4; j < dimensions; ++j) {
          if (data[j] > max_confidence) {
            max_confidence = data[j];
            class_id = j - 4;
          }
        }

        if (max_confidence > confidence_threshold) {
          // Extract bounding box coordinates
          float cx = data[0];
          float cy = data[1];
          float w = data[2];
          float h = data[3];

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
  cv::dnn::Net net_;
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
