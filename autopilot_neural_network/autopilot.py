import threading
import time

import torch
import PIL

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image

from autopilot_neural_network.parameters_client_node import ParametersClientNode
from scripts.model import AutopilotNet
from scripts.dataset import preprocessing_transform

from cv_bridge import CvBridge
import cv2


class Autopilot(ParametersClientNode):
    """
    Subscribes to camera images and continuously runs inference in a separate thread
    to generate steering and velocity commands. Older frames are dropped to ensure
    real-time processing. Uses parameters from a vehicle controller node to scale network 
    outputs to actual vehicle commands.
    """

    def __init__(self):
        """
        @brief Initializes the Autopilot node.

        Loads the neural network model, sets up image preprocessing, ROS publishers and subscribers,
        and starts the inference thread.
        """
        super().__init__("autopilot")

        # Load ROS parameters
        self.declare_parameter("vehicle_node", "vehicle_controller")
        self.declare_parameter("velocity_topic", "/velocity")
        self.declare_parameter("steering_angle_topic", "/steering_angle")
        self.declare_parameter("image_topic", "/camera/image_raw")
        
        self.declare_parameter("image_width", 128)
        self.declare_parameter("image_height", 96)
        
        self.declare_parameter('model_path', '/tmp/autopilot_neural_network/model.pt')
        
        self.declare_parameter("max_velocity_parameter", "max_velocity")
        self.declare_parameter("max_steering_angle_parameter", "max_steering_angle")
        
        # Retrieve parameters
        self.vehicle_node = self.get_parameter("vehicle_node").value
        self.velocity_topic = self.get_parameter("velocity_topic").value
        self.steering_angle_topic = self.get_parameter("steering_angle_topic").value
        self.image_topic = self.get_parameter("image_topic").value
        self.image_width = self.get_parameter("image_width").value
        self.image_height = self.get_parameter("image_height").value
        self.model_path = self.get_parameter("model_path").value
        
        max_velocity_parameter = self.get_parameter("max_velocity_parameter").value
        max_steering_angle_parameter = self.get_parameter("max_steering_angle_parameter").value

        # Vehicle controller parameters
        self.max_velocity = None
        self.max_steering_angle = None

        # Request vehicle controller parameters (max_velocity, max_steering_angle)
        self.max_velocity, self.max_steering_angle = self.request_parameters(
            self.vehicle_node, [max_velocity_parameter, max_steering_angle_parameter])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load neural network model
        self.get_logger().info(f"Loading model from: {self.model_path}")

        self.model = AutopilotNet()
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)

        # Image preprocessing
        self.bridge = CvBridge()
        
        # Initialize the image preprocessing pipeline to match the training configuration
        self.transform = preprocessing_transform(self.image_height, 
                                                 self.image_width)
        
        # QoS for subscriber: keep only the latest message
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                 history=QoSHistoryPolicy.KEEP_LAST,
                                 depth=1)

        # Publishers
        self.velocity_pub = self.create_publisher(
            Float64, self.velocity_topic, 1)
        
        self.steering_pub = self.create_publisher(
            Float64, self.steering_angle_topic, 1)

        # Subscriber
        self.create_subscription(Image, self.image_topic, 
                                 self._image_callback, qos_profile)

        # Inference management
        self.latest_frame = None # Stores the latest camera frame
        self.lock = threading.Lock() # Protects access to latest_frame
        self.running = True # Controls inference thread
        self.inference_thread = threading.Thread(target=self._inference_loop,
                                                 daemon=True)
        self.inference_thread.start()

        self.get_logger().info("Autopilot initialized")

    def _image_callback(self, msg: Image):
        """
        @brief Callback for incoming camera images.

        Stores the latest frame and discards older ones, implementing UDP-like behavior.

        @param msg ROS Image message from camera topic
        """
        with self.lock:
            self.latest_frame = msg

    def _inference_loop(self):
        """
        @brief Continuous loop that processes the latest frame.

        Runs in a separate thread, checking for new frames and running inference.
        Older frames are dropped to ensure real-time performance.
        """
        while self.running and rclpy.ok():
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None  # Drop older frames

            if frame is not None:
                self._process_frame(frame)
            else:
                time.sleep(0.005)  # Small delay to avoid busy-waiting

    def _process_frame(self, msg: Image):
        """
        @brief Converts the ROS Image message, runs inference, and publishes commands.

        Converts the image to grayscale, applies preprocessing, runs the neural network,
        scales outputs by vehicle parameters, and publishes velocity and steering messages.

        @param msg ROS Image message to process
        """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        gray_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        pil_gray = PIL.Image.fromarray(gray_image) # Convert to PIL Image
        input_tensor = self.transform(pil_gray).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_velocity = output[0, 0]
            pred_steering = output[0, 1]

        # Scale predictions to actual vehicle commands
        velocity = float(pred_velocity.item()) * self.max_velocity
        steering = float(pred_steering.item()) * self.max_steering_angle

        # Publish results
        self.velocity_pub.publish(Float64(data=velocity))
        self.steering_pub.publish(Float64(data=steering))

    def destroy_node(self):
        """
        @brief Stops the inference thread before destroying the ROS2 node.

        Ensures the inference thread is safely joined to avoid dangling threads.
        """
        self.running = False
        self.inference_thread.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = Autopilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
