# # Time of estimation calculation , and the coordinate frame ID
std_msgs/Header header

# Time of actual sensor data acquisition
float64 sourcetime

# Describes the channels

string[] fields

# How many traffic lights were detected
int8 count

# Actual traffic light 3D position data, size is (count * 9)
SinglePosition[] data