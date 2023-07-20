# custom_yolov8_ros [for ZED]

ROS 2 wrap for [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) to perform object detection and tracking.


## Installation
```shell
$ cd ~/your workspace/src
$ git clone https://github.com/mgonzs13/yolov8_ros.git
$ pip3 install -r yolov8_ros/requirements.txt
$ cd ~/your workspace
$ rosdep install --from-paths src --ignore-src -r -y
$ colcon build
```

## Usage
```shell
$ ros2 launch yolov8_bringup yolov8.launch.py
```

Thanks to [mgonzs13](https://github.com/mgonzs13/yolov8_ros)

