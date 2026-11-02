# Arduino sketches for the Jimny modules <img src="../docs/logos/spots_logo.png" alt="Spots Logo">

## General
The Arduino Nano boards used for the modules are older, using a USB mini B port. To use and flash these boards a special driver
is required. This `CH34x` driver located in this repo ensures the board can be flashed using the Arduino IDE.
If an error still occurs, please ensure the Old Bootloader is selected:
```
Tools > 
    Processor > 
        ATmega328P (Old Bootloader)
```

### CAN IDs
CAN messages have a specific ID. This ID allows nodes to determine if the message is meant for them and the priority in
which messages are sent. For this project we have settled on the following ID convention:

| CAN ID | Sender    | Purpose            | Data 1              | Data 2              |
|--------|-----------|--------------------|---------------------|---------------------|
| 0x01   | Jetson*   | Heartbeat          | Heartbeat interval  | Operating mode      |
| 0x11   | SteerBok  | Heartbeat/ logging | Steering angle      | Stepper rate output |
| 0x110  | Jetson*   | Steering target    | targetSteering      | targetSteeringRate  |
| 0x12   | AxelBrake | Heartbeat/ logging | Acceleration output |                     |
| 0x120  | Jetson*   | vel/acc target     | targetVelocity      | targetAcceleration  |
| 0x121  | Jetson*   | vel/acc actual     | actualGPSVelocity   | actualAcceleration  |

**While waiting for the Jetson CAN transceiver, the messages are currently sent by UART and put on the CAN bus by the Sterfboard*

It is important when constructing CAN messages to ensure the correct amount of bytes (4, 6 or 8 works). How many bytes
each `Data` block uses can be customized to improve data accuracy. Most of the data is sent using 2 bytes (16 bit).

### Heartbeat
To ensure each node is still running correctly it is required to send a heartbeat signal at a specified interval.
This heartbeat message is a CAN message that can contain data in the form of control signals or measurements. 
The node also expects to receive a heartbeat signal back at a set interval. If no heartbeat has been received the node
will go into a safety mode to stop the vehicle.

## sterfBoard

*For the [Jimny_power](https://gitlab.com/tut-robotics/spots/agv/jimny/-/tree/main/parts/PCBs/Jimny_power?ref_type=heads) PCB*

Code for the main power cutoff/inject board. 
### Functionality and Connectivity options

#### Jetson
This board can be connected to the Jetson to receive commands via CAN bus or UART/USB.
The PCB has a screw terminal to connect the CAN high and low signals. 
However, CAN bus is left as a future possibility since the required CAN transceiver for the Jetson is difficult to obtain.

The mini B Arduino Nanos however do not have a good supported USB driver for the Jetson, leaving this option undesired.
Therefore, we use UART communication between the Jetson and the sterfBoard to send Serial commands which the sterfBoard will convert to CAN bus.

#### CubePilot PWM
To allow the CubePilot to send a failsafe signal in case of an emergency, where the Jetson could be unresponsive. 
The PWM signal is filtered with an RC circuit and read by analog input A0.

#### I2C
The board has the I2C pins available at the screw terminal. 
This would in the future allow this module to directly send signals to the CubePilot in case an anomaly occurs and the
Jetson might be unresponsive. The CubePilot can then communicate back to the HereLink ground station that there is a problem.

These pins could also allow other I2C modules to be added to increase the functionality of the board.

#### Buttons
Upto 4 buttons can be connected to the board. The E button is meant as an emergency stop button, disabling the 48V relay
to disengage the stepper, and to send a emergency CAN heartbeat to the modules (mode=emergency) to let them know to stop.
This means the axelBrake should apply full braking.

The other 3 buttons are open for different usages, but currently configured as:
- autonomous mode enable
- power enable (turns on 48v and 9v power)
- arm/disarm

#### Relays
The board has two relays, used to turn on/off 9V power to all submodules and 48V to the stepper driver.

## axelBrake

*For the [Jimny_throttle](https://gitlab.com/tut-robotics/spots/agv/jimny/-/tree/main/parts/PCBs/Jimny_throttle?ref_type=heads) PCB*

Code for the acceleration and braking module. It receives velocity and acceleration setpoints and actual data from the 
GPS/IMU, to be used with an PID controller setting the required acceleration or braking.

## steerBok

*For the [Jimny_steering](https://gitlab.com/tut-robotics/spots/agv/jimny/-/tree/main/parts/PCBs/Jimny_steering?ref_type=heads) PCB*

Code for the steering module. It receives steering angle and rate setpoints. The actual steering angle is read using a
potentiometer mounted to the steering shaft. A PID controller is used to calculate the desired stepper pulse rate.
Using the `AccelStepper` library the stepper driver is controlled.
