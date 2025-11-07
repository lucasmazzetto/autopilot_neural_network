import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterValue, ParameterType
from typing import Any, List, Tuple


class ParametersClientNode(Node):
    """
    Base class for ROS 2 nodes that need to communicate with another node
    to retrieve parameters.

    This class simplifies the process of querying parameters from other ROS 2 nodes.
    It provides synchronous service calls to the GetParameters service and returns
    the requested values.
    """

    def __init__(self, node_name: str) -> None:
        """
        @brief Initializes the parameter client node.

        @param node_name Name to assign to this node in the ROS graph.
        """
        super().__init__(node_name)

    def request_parameters(self, target_node_name: str, param_names: List[str], 
                           timeout_sec: float = 2.0) -> Any | Tuple[Any, ...]:
        """
        @brief Synchronously requests parameters from a remote ROS 2 node.

        This method blocks until the GetParameters service responds or the
        specified timeout expires. Returned values are converted into native
        Python types and returned as a tuple in the same order as requested.

        @param target_node_name Name of the node providing the parameters.
        @param param_names List of parameter names to request.
        @param timeout_sec Maximum time to wait for the service response.

        @return Parameter value or tuple of parameter values.
        """
        # Create a client for the GetParameters service
        service_name = f"/{target_node_name}/get_parameters"
        client = self.create_client(GetParameters, service_name)

        # Wait for the service to become available
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError(f"GetParameters service not available: {service_name}")

        # Create the service request and set parameter names
        request = GetParameters.Request()
        request.names = param_names

        # Send the request asynchronously
        future = client.call_async(request)

        # Spin the node until the service response arrives or times out
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        # Fail if the service did not respond in time
        if not future.done():
            raise TimeoutError(f"Timeout while requesting parameters {param_names} "
                               f"from node '{target_node_name}'")

        # Get the result response
        response = future.result()
        
        # Validate response size matches requested parameters
        if response is None or len(response.values) != len(param_names):
            raise RuntimeError(f"Invalid parameter response from node '{target_node_name}'")

        # Convert ROS ParameterValue messages into native Python values
        values = tuple(self._extract_parameter_value(value) for value in response.values)

        # Return a single value or a tuple depending on request size
        return values[0] if len(values) == 1 else values

    def _extract_parameter_value(self, value: ParameterValue) -> Any:
        """
        @brief Converts a ROS 2 ParameterValue message into a Python-native value.

        @param value ParameterValue message returned by GetParameters.
        @return Corresponding Python value, or None if the type is unsupported.
        """
        match value.type:
            case ParameterType.PARAMETER_BOOL:
                return value.bool_value
            case ParameterType.PARAMETER_INTEGER:
                return value.integer_value
            case ParameterType.PARAMETER_DOUBLE:
                return value.double_value
            case ParameterType.PARAMETER_STRING:
                return value.string_value
            case ParameterType.PARAMETER_BYTE_ARRAY:
                return bytes(value.byte_array_value)
            case ParameterType.PARAMETER_INTEGER_ARRAY:
                return list(value.integer_array_value)
            case ParameterType.PARAMETER_DOUBLE_ARRAY:
                return list(value.double_array_value)
            case ParameterType.PARAMETER_BOOL_ARRAY:
                return list(value.bool_array_value)
            case ParameterType.PARAMETER_STRING_ARRAY:
                return list(value.string_array_value)
            case _:
                self.get_logger().warn(f"Unknown parameter type received: {value.type}")
                return None
