import os
import rclpy
import csv
import cv2

from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from autopilot_neural_network.parameters_client_node import ParametersClientNode
from dataclasses import dataclass
from cv_bridge import CvBridge


@dataclass
class VehicleState:
    steering_angle = None
    velocity = None
    image = None
    steering_angle_time = None
    velocity_time = None
    image_time = None
    

class CsvDataStorage:
    def __init__(self, dataset_path: str):
        """
        @brief Initializes the CSV data storage handler.

        Creates the dataset directory if it does not exist, and prepares the CSV file path.

        @param dataset_path Path to the dataset directory where images and CSV files are stored.
        """
        self.dataset_path = dataset_path
        self.csv_file_path = os.path.join(self.dataset_path, 'dataset.csv')
        self.bridge = CvBridge()

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

    def _get_next_image_id(self) -> int:
        """
        @brief Retrieves the next available image ID for saving a new image.

        Scans the dataset directory for existing images and determines the next sequential ID.

        @return The next image ID as an integer.
        """
        os.makedirs(self.dataset_path, exist_ok=True)  # Ensure directory exists

        existing_images = [f for f in os.listdir(self.dataset_path) if f.endswith('.png')]
        if not existing_images:
            return 0

        existing_ids = [int(os.path.splitext(f)[0]) for f in existing_images if f.split('.')[0].isdigit()]
        
        return max(existing_ids) + 1

    def _save_image(self, image_id: int, image: Image) -> str:
        """
        @brief Saves a ROS Image message as a PNG file.

        Converts the ROS Image message to an OpenCV image and writes it to the dataset directory.

        @param image_id The numeric ID of the image to save.
        @param image The Image message received from ROS.
        @return The full path to the saved image file.
        """
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        image_path = os.path.join(self.dataset_path, f'{image_id}.png')
        cv2.imwrite(image_path, cv_image)
        return image_path

    def _append_csv_row(self, image_id: int, state: VehicleState, 
                        max_velocity: float, max_steering_angle: float): 
        """
        @brief Appends a new row of vehicle state data to the dataset CSV file.

        If the CSV file does not exist yet, a header row is written before appending data.

        @param image_id The ID associated with the image for this data sample.
        @param state The VehicleState object containing the state of the vehicle.
        """
        write_header = not os.path.exists(self.csv_file_path) 
        
        with open(self.csv_file_path, 'a', newline='') as csvfile: 
            writer = csv.writer(csvfile) 
            
            if write_header: 
                writer.writerow(['image_id', 'velocity', 'steering_angle', 
                                 'image_time', 'velocity_time', 'steering_angle_time',
                                 'max_velocity', 'max_steering_angle']) 
                
            writer.writerow([image_id, 
                             state.velocity, 
                             state.steering_angle, 
                             state.image_time.nanoseconds / 1e9, # To UNIX timestamp
                             state.velocity_time.nanoseconds / 1e9, # To UNIX timestamp
                             state.steering_angle_time.nanoseconds / 1e9, # To UNIX timestamp
                             max_velocity, 
                             max_steering_angle])

    def save_sample(self, state: VehicleState, max_velocity: float, max_steering_angle: float):
        """
        @brief Saves a complete sample, including the image and corresponding state data.

        This method assigns an image ID, saves the image file, and appends metadata to the CSV file.

        @param state The VehicleState object containing vehicle state data.
        """
        image_id = self._get_next_image_id()
        self._save_image(image_id, state.image)
        self._append_csv_row(image_id, state, max_velocity, max_steering_angle)
    
    
class DataCollector(ParametersClientNode):
    """
    This node subscribes to vehicle-related topics, retrieves parameters from the 
    vehicle controller, and periodically saves synchronized image and state data to a dataset.
    """
        
    def __init__(self):
        super().__init__("data_collector")

        # Load ROS parameters
        self.declare_parameter("min_velocity_factor", 0.25)
        self.declare_parameter("timeout_time", 1_000_000_000)
        self.declare_parameter("update_period", 0.5)
        self.declare_parameter("dataset_path", "/tmp/autopilot_neural_network/dataset")
        
        self.declare_parameter("vehicle_node", "vehicle_controller")
        self.declare_parameter("velocity_topic", "vehicle_controller")
        self.declare_parameter("steering_angle_topic", "vehicle_controller")
        self.declare_parameter("image_topic", "vehicle_controller")
        
        self.declare_parameter("max_velocity_parameter", "max_velocity")
        self.declare_parameter("max_steering_angle_parameter", "max_steering_angle")

        # Retrieve parameters
        self.min_velocity_factor = self.get_parameter("min_velocity_factor").value
        self.timeout_time = self.get_parameter("timeout_time").value
        self.update_period = self.get_parameter("update_period").value
        self.dataset_path = self.get_parameter("dataset_path").value
       
        self.vehicle_node = self.get_parameter("vehicle_node").value
        self.velocity_topic = self.get_parameter("velocity_topic").value
        self.steering_angle_topic = self.get_parameter("steering_angle_topic").value
        self.image_topic = self.get_parameter("image_topic").value
        
        max_velocity_parameter = self.get_parameter("max_velocity_parameter").value
        max_steering_angle_parameter = self.get_parameter("max_steering_angle_parameter").value

        # Vehicle controller parameters
        self.max_velocity = None
        self.max_steering_angle = None

        # Request vehicle controller parameters (max_velocity, max_steering_angle)
        self.max_velocity, self.max_steering_angle = self.request_parameters(
            self.vehicle_node, [max_velocity_parameter, max_steering_angle_parameter])

        # Storage layer
        self.storage = CsvDataStorage(self.dataset_path)
        self.last_state = VehicleState()

        # QoS: keep only the latest message, replacing older ones
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                 history=QoSHistoryPolicy.KEEP_LAST,
                                 depth=1)

        # Subscribers
        self.create_subscription(Float64, 
                                 self.steering_angle_topic, 
                                 self._steering_callback, qos_profile)
        
        self.create_subscription(Float64, 
                                 self.velocity_topic, 
                                 self._velocity_callback, qos_profile)
        
        self.create_subscription(Image, 
                                 self.image_topic, 
                                 self._image_callback, qos_profile)

        # Timer for periodic global loop execution
        self.timer = self.create_timer(self.update_period, self._update)

    def _has_invalid_state(self) -> bool:
        """
        @brief Checks if the current state is invalid.

        Evaluates timeout, missing data, or low-speed conditions to determine
        if the current vehicle state is invalid for recording.

        @retval True If any invalid condition is met (timeout, missing data, or low speed).
        @retval False If the current state is valid.
        """
        now = self.get_clock().now()

        # Missing data check (return immediately)
        required_fields = (
            self.last_state.image,
            self.last_state.image_time,
            self.last_state.steering_angle,
            self.last_state.steering_angle_time,
            self.last_state.velocity,
            self.last_state.velocity_time,
            self.max_velocity,
            self.max_steering_angle,
        )

        if any(value is None for value in required_fields):
            return True

        # Timeout check (safe because timestamps are guaranteed valid if we reached here)
        timestamps = [
            self.last_state.image_time,
            self.last_state.steering_angle_time,
            self.last_state.velocity_time,
        ]

        if (now - min(timestamps)).nanoseconds > self.timeout_time:
            return True

        # Low-speed check
        if (self.max_velocity is not None and
            self.last_state.velocity <
            self.min_velocity_factor * self.max_velocity):
            return True

        # All good
        return False
        
    def _steering_callback(self, msg: Float64):
        """
        @brief Callback for receiving steering angle data.

        @param msg Float64 message containing the current steering angle.
        """
        self.last_state.steering_angle = msg.data
        self.last_state.steering_angle_time = self.get_clock().now()

    def _velocity_callback(self, msg: Float64):
        """
        @brief Callback for receiving vehicle velocity data.

        @param msg Float64 message containing the current vehicle velocity.
        """
        self.last_state.velocity = msg.data
        self.last_state.velocity_time = self.get_clock().now()

    def _image_callback(self, msg: Image):
        """
        @brief Callback for receiving camera images.

        @param msg Image message from the camera topic.
        """
        self.last_state.image = msg
        self.last_state.image_time = self.get_clock().now()

    def _update(self):
        """
        @brief Periodic update loop executed by the node's timer.

        Checks the validity of the current vehicle state and, if valid,
        saves a synchronized sample (image + state data) to the dataset.
        """
        if self._has_invalid_state():
            return

        self.storage.save_sample(self.last_state, self.max_velocity, self.max_steering_angle) 

        self.get_logger().info(f"Velocity: {self.last_state.velocity}, " 
                               f"Steering Angle: {self.last_state.steering_angle}")


def main(args=None):
    rclpy.init(args=args)

    # Instantiate the node with the given vehicle node name
    node = DataCollector()

    try:
        # Spin the node to process callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
