std_msgs/Header header
int16 rssi
int16[] channels
# Raw PWM (us) of every channel the Cube reports (index 0 = channel 1).
# 65535 means the channel is unused.
uint16[] channels_raw
