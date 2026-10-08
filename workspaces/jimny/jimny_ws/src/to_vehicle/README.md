# `to_vehicle` package

Sends the `Ackermann_msg` to the vehicle and receives data back from the vehicle. 
Communication is either via serial/UART (for the Rover) or direct CAN bus (for the Jimny).


---

## Serial mode
The Jetson will connect to a serially connected microcontroller to send the vehicle commands.
Message is sent with a standard format of:
```
<{set_steering_angle}, {set_steering_angle_velocity}, {set_velocity}, {set_acceleration}, {set_jerk}, {current_gps_speed}>
```
The microcontroller is expected to send back a heartbeat at 1Hz to show it is still alive. Similarly, the microcontroller can 
expect to receive a heartbeat message with a frequency of 1 Hz. 

### Rover
In the rover folder the *.ino* file can be found containing the ESP32 code for the Rover.

### Jimny
In the jimny folder the *.ino* file can be found containing the Arduino nano code to receive the message and resend it over CAN bus.
This is a temporary solution until the correct CAN transceiver is ordered/received to directly use the Jetson CAN controller. 

## CAN mode
Currently this mode is not tested
