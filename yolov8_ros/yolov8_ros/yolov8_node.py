import cv2
import torch
import random
import tensorrt

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import SetBool


class Yolov8Node(Node):
    def __init__(self) -> None:
        super().__init__("yolov8_node")

        # params
        self.declare_parameter("model", "yolov8m.pt")
        model = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("tracker", "bytetrack.yaml")
        tracker = self.get_parameter("tracker").get_parameter_value().string_value

        self.declare_parameter("device", "cpu")  # "cuda:0"
        device = self.get_parameter("device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.6)
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )

        self.declare_parameter("enable", True)
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value

        self._class_to_color = {}
        self.cv_bridge = CvBridge()
        self.tracker = self.create_tracker(tracker)
        self.yolo = YOLO(model, task='detect')
        if model[-2:] == "pt":
            self.yolo.fuse()
            self.yolo.to(device)

        # topcis
        self._front_detection_pub = self.create_publisher(Detection2DArray, "/yolo/front_detections", 10)
        self._front_image_pub = self.create_publisher(Image, "/yolo/front_image", 10)
        self._down_detection_pub = self.create_publisher(Detection2DArray, "/yolo/down_detections", 10)
        self._down_image_pub = self.create_publisher(Image, "/yolo/down_image", 10)
        self._sub1 = self.create_subscription(
            Image,
            "image_raw",
            self.front_image_cb,  # Image, "/camera/color/image_raw", self.image_cb,
            qos_profile_sensor_data,
        )
        self._sub2 = self.create_subscription(
            Image,
            "gopro/image_raw",
            self.down_image_cb,  # Image, "/camera/color/image_raw", self.image_cb,
            qos_profile_sensor_data,
        )

        # services
        self._srv = self.create_service(SetBool, "enable", self.enable_cb)

    def create_tracker(self, tracker_yaml) -> BaseTrack:
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in [
            "bytetrack",
            "botsort",
        ], f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def enable_cb(
        self, req: SetBool.Request, res: SetBool.Response
    ) -> SetBool.Response:
        self.enable = req.data
        res.success = True
        return res

    def front_image_cb(self, msg: Image) -> None:
        if self.enable:
            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            model = self.get_parameter("model").get_parameter_value().string_value
            if model[-2] == "pt":
                results = self.yolo.predict(
                    source=cv_image, verbose=False, stream=False, conf=0.4, max_det=3, classes=[0,2]
                )
            else:
                results = self.yolo.predict(
                    source=cv_image,
                    verbose=False,
                    stream=False,
                    conf=0.4,
                    half=True,
                    device=0,
                    max_det=3,
                    classes=[0,2]
                )

            # track
            det = results[0].boxes.cpu().numpy()

            if len(det) > 0:
                im0s = self.yolo.predictor.batch[2]
                im0s = im0s if isinstance(im0s, list) else [im0s]

                tracks = self.tracker.update(det, im0s[0])
                if len(tracks) > 0:
                    results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

            # create detections msg
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            results = results[0].cpu()

            for b in results.boxes:
                try:
                    label = self.yolo.names[int(b.cls)]
                except:
                    names = ["Cross", "Helipad", "Ladder"]
                    label = names[int(b.cls)]

                score = float(b.conf)

                if score < self.threshold:
                    continue

                detection = Detection2D()

                box = b.xywh[0]

                # get boxes values
                detection.bbox.center.x = float(
                    box[0]
                )  # detection.bbox.center.position.x
                detection.bbox.center.y = float(box[1])
                detection.bbox.size_x = float(box[2])
                detection.bbox.size_y = float(box[3])

                # get track id
                track_id = -1
                if not b.id is None:
                    track_id = int(b.id)
                # detection.id = str(track_id)

                # get hypothesis
                hypothesis = ObjectHypothesisWithPose()
                # hypothesis.hypothesis.class_id = label
                hypothesis.id = label
                # hypothesis.hypothesis.score = score
                hypothesis.score = score
                detection.results.append(hypothesis)

                # draw boxes for debug
                if label not in self._class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    self._class_to_color[label] = (r, g, b)
                color = self._class_to_color[label]

                min_pt = (
                    round(
                        detection.bbox.center.x - detection.bbox.size_x / 2.0
                    ),  # round(detection.bbox.center.position.x - detection.bbox.size_x / 2.0
                    round(detection.bbox.center.y - detection.bbox.size_y / 2.0),
                )
                max_pt = (
                    round(detection.bbox.center.x + detection.bbox.size_x / 2.0),
                    round(detection.bbox.center.y + detection.bbox.size_y / 2.0),
                )
                cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

                label = "{} ({:.3f})".format(label, score)
                pos = (min_pt[0] + 5, min_pt[1] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_image, label, pos, font, 1, color, 1, cv2.LINE_AA)

                # append msg
                detections_msg.detections.append(detection)

            # publish detections and dbg image
            self._front_detection_pub.publish(detections_msg)
            self._front_image_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(cv_image, encoding=msg.encoding)
            )
        cv2.imshow("front_result", cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(10)

    def down_image_cb(self, msg: Image) -> None:
        if self.enable:
            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            model = self.get_parameter("model").get_parameter_value().string_value
            if model[-2] == "pt":
                results = self.yolo.predict(
                    source=cv_image, verbose=False, stream=False, conf=0.4, max_det=1, classes=1
                )
            else:
                results = self.yolo.predict(
                    source=cv_image,
                    verbose=False,
                    stream=False,
                    conf=0.4,
                    half=True,
                    device=0,
                    max_det=1,
                    classes=1
                )

            # track
            det = results[0].boxes.cpu().numpy()

            if len(det) > 0:
                im0s = self.yolo.predictor.batch[2]
                im0s = im0s if isinstance(im0s, list) else [im0s]

                tracks = self.tracker.update(det, im0s[0])
                if len(tracks) > 0:
                    results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

            # create detections msg
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            results = results[0].cpu()

            for b in results.boxes:
                try:
                    label = self.yolo.names[int(b.cls)]
                except:
                    names = ["Cross", "Helipad", "Ladder"]
                    label = names[int(b.cls)]

                score = float(b.conf)

                if score < self.threshold:
                    continue

                detection = Detection2D()

                box = b.xywh[0]

                # get boxes values
                detection.bbox.center.x = float(
                    box[0]
                )  # detection.bbox.center.position.x
                detection.bbox.center.y = float(box[1])
                detection.bbox.size_x = float(box[2])
                detection.bbox.size_y = float(box[3])

                # get track id
                track_id = -1
                if not b.id is None:
                    track_id = int(b.id)
                # detection.id = str(track_id)

                # get hypothesis
                hypothesis = ObjectHypothesisWithPose()
                # hypothesis.hypothesis.class_id = label
                hypothesis.id = label
                # hypothesis.hypothesis.score = score
                hypothesis.score = score
                detection.results.append(hypothesis)

                # draw boxes for debug
                if label not in self._class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    self._class_to_color[label] = (r, g, b)
                color = self._class_to_color[label]

                min_pt = (
                    round(
                        detection.bbox.center.x - detection.bbox.size_x / 2.0
                    ),  # round(detection.bbox.center.position.x - detection.bbox.size_x / 2.0
                    round(detection.bbox.center.y - detection.bbox.size_y / 2.0),
                )
                max_pt = (
                    round(detection.bbox.center.x + detection.bbox.size_x / 2.0),
                    round(detection.bbox.center.y + detection.bbox.size_y / 2.0),
                )
                cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

                label = "{} ({:.3f})".format(label, score)
                pos = (min_pt[0] + 5, min_pt[1] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_image, label, pos, font, 1, color, 1, cv2.LINE_AA)

                # append msg
                detections_msg.detections.append(detection)

            # publish detections and dbg image
            self._down_detection_pub.publish(detections_msg)
            self._down_image_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(cv_image, encoding=msg.encoding)
            )
        cv2.imshow("down_result", cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(10)


def main():
    rclpy.init()
    node = Yolov8Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()
