# etrobo_object_detection

A ROS2 package for real-time object detection using YOLOv8 with ONNX Runtime and NCNN backends, designed for ET Robocon applications.

## Features

- **YOLOv8 Object Detection**: High-performance object detection using YOLOv8 models
- **Dual Backend Support**: 
  - **ONNX Runtime**: Cross-platform optimized inference with ONNX Runtime C++ API
  - **NCNN**: High-performance mobile inference framework with Vulkan support
- **ROS2 Native**: Full ROS2 parameter support and standard message interfaces
- **Configurable Parameters**: Flexible configuration for different use cases
- **Real-time Performance**: Optimized for real-time robot applications

## Requirements

### System Requirements
- Ubuntu 22.04
- ROS 2 Humble
- OpenCV 4.5+
- Python 3.10 (for model conversion)

### Dependencies
- `rclcpp`
- `sensor_msgs`
- `vision_msgs`
- `cv_bridge`
- `image_transport` 
- `opencv4`

#### For ONNX Runtime Backend
- ONNX Runtime 1.17.3 (automatically downloaded during build)

#### For NCNN Backend  
- NCNN library (build and install from source)
- OpenMP (`sudo apt install libomp-dev`)

### Supported Architectures
- **x86_64** (Intel/AMD 64-bit)
- **aarch64** (ARM64, including Raspberry Pi 4/5)

The build system automatically detects your architecture and downloads the appropriate ONNX Runtime binary.

## Installation

### 1. Clone the Repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/owhinata/etrobo_object_detection.git
```

### 2. Install Dependencies
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Build the Package
```bash
# ONNX Runtime will be automatically downloaded during the first build
colcon build --symlink-install --packages-select etrobo_object_detection
source install/setup.bash
```

**Note**: The first build will take longer as it downloads ONNX Runtime (~100MB for your architecture). The build system automatically detects whether you're on x86_64 or aarch64 and downloads the appropriate binary. Subsequent builds will be faster.

## Model Preparation

This package supports two inference backends with different model formats:

### ONNX Runtime Backend

#### Download YOLOv8 ONNX Model
```bash
# Install ultralytics (if not already installed)
pip install ultralytics

# Download and convert YOLOv8 model to ONNX
# Input size is automatically detected from the model, so choose based on your performance needs:

# For faster inference (recommended for real-time applications):
yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12 dynamic=False

# For balanced performance:
yolo export model=yolov8n.pt format=onnx imgsz=480 opset=12 dynamic=False

# For higher accuracy:
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12 dynamic=False

# Alternative model sizes (optional):
# yolo export model=yolov8s.pt format=onnx imgsz=320 opset=12 dynamic=False
# yolo export model=yolov8m.pt format=onnx imgsz=320 opset=12 dynamic=False
# yolo export model=yolov8l.pt format=onnx imgsz=320 opset=12 dynamic=False
# yolo export model=yolov8x.pt format=onnx imgsz=320 opset=12 dynamic=False
```

**Important Parameters:**
- `imgsz=320/480/640`: Input image size (automatically detected by the application)
- `opset=12`: ONNX opset version for compatibility
- `dynamic=False`: Fixed input shape for optimized inference

**Note**: The application automatically detects the input size from the loaded ONNX model, so you don't need to configure it manually.

### Model Quantization (Optional)

For better performance and smaller model size, you can quantize the ONNX model to INT8:

```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_model():
    model_fp32 = "yolov8n.onnx"
    model_int8 = "yolov8n_int8.onnx"

    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_int8,
        weight_type=QuantType.QUInt8
    )

    # サイズ比較
    fp32_size = os.path.getsize(model_fp32) / 1024 / 1024
    int8_size = os.path.getsize(model_int8) / 1024 / 1024

    print(f"FP32: {fp32_size:.1f}MB → INT8: {int8_size:.1f}MB")
    print(f"圧縮率: {fp32_size/int8_size:.1f}x")

if __name__ == "__main__":
    quantize_model()
```

Run the script to create a quantized model:
```bash
python quantize_model.py
```

The quantized model typically provides:
- **~4x smaller file size**
- **Faster inference speed**
- **Minimal accuracy loss** (usually <2%)

#### Place ONNX Model File
Copy the generated `yolov8n.onnx` (or `yolov8n_int8.onnx` for quantized version) file to your workspace or specify the full path using parameters.

### NCNN Backend

#### Download and Convert YOLOv8 Model for NCNN

For NCNN backend, you need to convert YOLOv8 to NCNN format following these steps:

```bash
# 1. Install required tools
pip3 install -U ultralytics pnnx ncnn

# 2. Export YOLOv8 to TorchScript
yolo export model=yolov8n.pt format=torchscript

# 3. Convert TorchScript with static shape
pnnx yolov8n.torchscript

# 4. Modify yolov8n_pnnx.py for dynamic shape inference
# Edit the generated yolov8n_pnnx.py file to modify tensor operations:
# 
# Before:
#     v_165 = v_142.view(1, 144, 6400)
#     v_166 = v_153.view(1, 144, 1600)
#     v_167 = v_164.view(1, 144, 400)
#     v_168 = torch.cat((v_165, v_166, v_167), dim=2)
#     ...
# 
# After:
#     v_165 = v_142.view(1, 144, -1).transpose(1, 2)
#     v_166 = v_153.view(1, 144, -1).transpose(1, 2)
#     v_167 = v_164.view(1, 144, -1).transpose(1, 2)
#     v_168 = torch.cat((v_165, v_166, v_167), dim=1)
#     return v_168

# 5. Re-export YOLOv8 TorchScript with modifications
python3 -c 'import yolov8n_pnnx; yolov8n_pnnx.export_torchscript()'

# 6. Convert new TorchScript with dynamic shape support
pnnx yolov8n_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]

# 7. Rename output files to final NCNN model files
mv yolov8n_pnnx.py.ncnn.param yolov8n.ncnn.param
mv yolov8n_pnnx.py.ncnn.bin yolov8n.ncnn.bin
```

**Important Notes**:
- The modification in step 4 is crucial for dynamic shape inference
- You need to manually edit the generated Python file to change tensor operations
- The final output will be a 2-dim tensor with dimensions [144, 8400] containing bounding box regression (16x4) and per-class scores (80 classes)

#### Place NCNN Model Files
Copy the generated `yolov8n.ncnn.param` and `yolov8n.ncnn.bin` files to your workspace or specify the full paths using parameters.

## Usage

### Basic Usage

#### ONNX Runtime Backend
```bash
# Run with default parameters
ros2 run etrobo_object_detection etrobo_object_detection

# Run with custom parameters
ros2 run etrobo_object_detection etrobo_object_detection \
  --ros-args \
  -p model_path:=/path/to/yolov8s.onnx \
  -p confidence_threshold:=0.3 \
  -p input_topic:=/camera/image_raw
```

#### NCNN Backend
```bash
# Run with default parameters
ros2 run etrobo_object_detection etrobo_object_detection_ncnn

# Run with custom parameters  
ros2 run etrobo_object_detection etrobo_object_detection_ncnn \
  --ros-args \
  -p param_path:=/path/to/yolov8n.ncnn.param \
  -p bin_path:=/path/to/yolov8n.ncnn.bin \
  -p confidence_threshold:=0.3 \
  -p use_vulkan:=true
```

### Using Parameter File

#### ONNX Runtime Parameter File
Create a parameter file `config_onnx.yaml`:
```yaml
etrobo_object_detection:
  ros__parameters:
    model_path: "/path/to/yolov8n.onnx"
    confidence_threshold: 0.5
    nms_threshold: 0.4
    num_threads: 2
    input_topic: "/image_raw"
    output_topic: "/object_detection/image/compressed"
```

#### NCNN Parameter File
Create a parameter file `config_ncnn.yaml`:
```yaml
object_detection_ncnn:
  ros__parameters:
    param_path: "/path/to/yolov8n.ncnn.param"
    bin_path: "/path/to/yolov8n.ncnn.bin"
    confidence_threshold: 0.5
    nms_threshold: 0.4
    num_threads: 2
    input_topic: "/image_raw"
    output_topic: "/object_detection/image/compressed"
    use_vulkan: true
```

Run with parameter file:
```bash
# ONNX Runtime
ros2 run etrobo_object_detection etrobo_object_detection \
  --ros-args --params-file config_onnx.yaml

# NCNN  
ros2 run etrobo_object_detection etrobo_object_detection_ncnn \
  --ros-args --params-file config_ncnn.yaml
```

### Launch File Example
Create a launch file `detection.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='etrobo_object_detection',
            executable='etrobo_object_detection',
            name='object_detection_node',
            parameters=[{
                'model_path': '/path/to/yolov8n.onnx',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'num_threads': 2,
                'input_topic': '/camera/image_raw',
                'output_topic': '/object_detection/image/compressed'
            }]
        )
    ])
```

## Parameters

### ONNX Runtime Backend Parameters

#### High Priority Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `"yolov8n.onnx"` | Path to YOLOv8 ONNX model file |
| `confidence_threshold` | double | `0.5` | Minimum confidence score for detections |
| `nms_threshold` | double | `0.4` | Non-Maximum Suppression threshold |
| `num_threads` | int | `2` | Number of threads for ONNX Runtime inference |

#### Medium Priority Parameters  
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | `"/image_raw"` | Input image topic name |
| `output_topic` | string | `"/object_detection/image/compressed"` | Output compressed image topic name |

### NCNN Backend Parameters

#### High Priority Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_path` | string | `"yolov8n.ncnn.param"` | Path to NCNN parameter file |
| `bin_path` | string | `"yolov8n.ncnn.bin"` | Path to NCNN binary file |
| `confidence_threshold` | double | `0.5` | Minimum confidence score for detections |
| `nms_threshold` | double | `0.4` | Non-Maximum Suppression threshold |
| `num_threads` | int | `2` | Number of threads for NCNN inference |

#### Medium Priority Parameters  
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | `"/image_raw"` | Input image topic name |
| `output_topic` | string | `"/object_detection/image/compressed"` | Output compressed image topic name |
| `use_vulkan` | bool | `true` | Enable Vulkan GPU acceleration |

**Note**: 
- Input size is automatically handled by the respective inference engines.
- Output images are published only when there are subscribers to the output topic.
- Images are compressed to JPEG format (quality 80%) to reduce bandwidth usage.

## Topics

### Subscribed Topics
- `{input_topic}` (`sensor_msgs/Image`): Input camera images

### Published Topics
- `{output_topic}` (`sensor_msgs/CompressedImage`): Detection result images with bounding boxes (JPEG compressed)
- `/object_detection/detections` (`vision_msgs/Detection2DArray`): Object detection results with bounding boxes, class IDs, and confidence scores

### Detection Results Format

The `/object_detection/detections` topic publishes detection results in the standard `vision_msgs/Detection2DArray` format, which includes:

#### Message Structure
```yaml
header:
  stamp:     # Timestamp from input image (preserves temporal accuracy)
  frame_id:  # Frame ID from input image (or "camera_frame" if empty)
detections:  # Array of detected objects
  - bbox:    # Bounding box information
      center:
        position:
          x: 320.5    # Center X coordinate (pixels)
          y: 240.0    # Center Y coordinate (pixels)
        theta: 0.0    # Rotation (always 0 for axis-aligned boxes)
      size_x: 150.0   # Bounding box width (pixels)
      size_y: 200.0   # Bounding box height (pixels)
    results:           # Classification results
      - hypothesis:
          class_id: "0"      # COCO class ID (string format)
          score: 0.85        # Confidence score (0.0-1.0)
```

#### COCO Class IDs
The detection results use COCO dataset class IDs (0-79):
- `0`: person, `1`: bicycle, `2`: car, `3`: motorcycle, `4`: airplane
- `5`: bus, `6`: train, `7`: truck, `8`: boat, `9`: traffic light
- ... (see [COCO classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) for full list)

#### Key Features
- **Always Published**: Empty arrays are sent when no objects are detected
- **Temporal Accuracy**: Uses original image timestamp, not processing completion time
- **Standard Format**: Compatible with other ROS2 vision packages
- **Complete Information**: Includes bounding boxes, confidence scores, and class labels

### Usage Examples

#### Subscribe to Detection Results
```bash
# View all detection messages
ros2 topic echo /object_detection/detections

# View detection headers only
ros2 topic echo /object_detection/detections --field header

# View detection count
ros2 topic echo /object_detection/detections --field "len(detections)"
```

#### Process Detection Results in Code
```python
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class DetectionSubscriber(Node):
    def __init__(self):
        super().__init__('detection_subscriber')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/object_detection/detections',
            self.detection_callback,
            10)

    def detection_callback(self, msg):
        self.get_logger().info(f'Received {len(msg.detections)} detections')
        
        for detection in msg.detections:
            # Extract bounding box
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y
            
            # Extract classification
            if detection.results:
                class_id = detection.results[0].hypothesis.class_id
                confidence = detection.results[0].hypothesis.score
                
                self.get_logger().info(
                    f'Object: class_id={class_id}, confidence={confidence:.2f}, '
                    f'bbox=({center_x:.1f},{center_y:.1f},{width:.1f},{height:.1f})'
                )
```

## Testing

### Unit Tests
```bash
colcon test --packages-select etrobo_object_detection
```

### Integration Testing
1. **Camera Integration**:
   ```bash
   # Terminal 1: Start camera
   ros2 run usb_cam usb_cam_node_exe
   
   # Terminal 2: Start detection
   ros2 run etrobo_object_detection etrobo_object_detection
   ```

2. **Simulation Testing**:
   ```bash
   # With etrobo_simulator
   ros2 launch etrobo_simulator simulation.launch.py
   ros2 run etrobo_object_detection etrobo_object_detection \
     --ros-args -p input_topic:=/camera/image_raw
   
   # View detection results
   ros2 run rqt_image_view rqt_image_view /object_detection/image/compressed
   
   # View detection data
   ros2 topic echo /object_detection/detections
   ```

## Performance Optimization

### Model Selection
- **YOLOv8n**: Fastest, suitable for real-time applications
- **YOLOv8s**: Balanced speed and accuracy
- **YOLOv8m/l/x**: Higher accuracy but slower

### Runtime Optimization
- Adjust confidence threshold based on use case
- Consider input resolution vs. performance trade-offs
- Tune `num_threads` parameter based on your CPU cores for optimal performance
- Monitor CPU/GPU usage and adjust accordingly

## Troubleshooting

### Common Issues

1. **Model Loading Error**:
   ```
   ONNX Runtime error: Model loading failed
   ```
   - Check model path is correct
   - Ensure model file is valid ONNX format
   - Verify file permissions

2. **No Image Received**:
   ```
   No images on input topic
   ```
   - Check topic name: `ros2 topic list`
   - Verify camera is publishing: `ros2 topic echo /image_raw`
   - Check QoS compatibility

3. **Low Detection Performance**:
   - Adjust `confidence_threshold` (lower = more detections)
   - Check model compatibility with your objects
   - Verify image quality and lighting

### Debug Mode
Enable detailed logging:
```bash
ros2 run etrobo_object_detection etrobo_object_detection \
  --ros-args --log-level DEBUG
```

## Development

### Code Structure
```
src/
├── etrobo_object_detection_onnx.cpp  # ONNX Runtime backend implementation
└── etrobo_object_detection_ncnn.cpp  # NCNN backend implementation
```

### Adding New Features
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Follow coding guidelines (PEP 8 style for consistency)
4. Add tests for new functionality
5. Submit pull request

### Pull Request Requirements
- **Requirement**: Copy task details verbatim
- **Change Summary**: Describe what was changed
- **Testing**: Detail testing performed

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please read the development guidelines above and ensure all tests pass before submitting pull requests.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing GitHub issues  
3. Create a new issue with detailed information

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform optimized inference
- [NCNN](https://github.com/Tencent/ncnn) for high-performance mobile inference framework
- ET Robocon community for requirements and testing
