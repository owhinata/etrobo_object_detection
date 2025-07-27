# etrobo_object_detection

A ROS2 package for real-time object detection using YOLOv8 and ONNX Runtime, designed for ET Robocon applications.

## Features

- **YOLOv8 Object Detection**: High-performance object detection using YOLOv8 models
- **ONNX Runtime Integration**: Optimized inference with ONNX Runtime C++ API
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
- `cv_bridge`
- `image_transport` 
- `opencv4`
- ONNX Runtime 1.17.3 (automatically downloaded during build)

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
colcon build --packages-select etrobo_object_detection
source install/setup.bash
```

**Note**: The first build will take longer as it downloads ONNX Runtime (~100MB for your architecture). The build system automatically detects whether you're on x86_64 or aarch64 and downloads the appropriate binary. Subsequent builds will be faster.

## Model Preparation

### Download YOLOv8 Model
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

### Place Model File
Copy the generated `yolov8n.onnx` (or `yolov8n_int8.onnx` for quantized version) file to your workspace or specify the full path using parameters.

## Usage

### Basic Usage
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

### Using Parameter File
Create a parameter file `config.yaml`:
```yaml
etrobo_object_detection:
  ros__parameters:
    model_path: "/path/to/yolov8n.onnx"
    confidence_threshold: 0.5
    nms_threshold: 0.4
    num_threads: 2
    input_topic: "/image_raw"
    display_results: true
```

Run with parameter file:
```bash
ros2 run etrobo_object_detection etrobo_object_detection \
  --ros-args --params-file config.yaml
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
                'display_results': True
            }]
        )
    ])
```

## Parameters

### High Priority Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `"yolov8n.onnx"` | Path to YOLOv8 ONNX model file |
| `confidence_threshold` | double | `0.5` | Minimum confidence score for detections |
| `nms_threshold` | double | `0.4` | Non-Maximum Suppression threshold |
| `num_threads` | int | `2` | Number of threads for ONNX Runtime inference |

### Medium Priority Parameters  
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | `"/image_raw"` | Input image topic name |
| `display_results` | bool | `true` | Enable/disable result visualization |

**Note**: Input size is automatically detected from the ONNX model - no manual configuration required.

## Topics

### Subscribed Topics
- `{input_topic}` (`sensor_msgs/Image`): Input camera images

### Published Topics
Currently, this package focuses on visualization. Publishing detection results as ROS messages can be added as needed.

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
├── etrobo_object_detection_onnx.cpp  # Main C++ implementation
└── etrobo_object_detection.cpp       # Legacy OpenCV DNN version
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
- [ONNX Runtime](https://onnxruntime.ai/) for optimized inference
- ET Robocon community for requirements and testing
