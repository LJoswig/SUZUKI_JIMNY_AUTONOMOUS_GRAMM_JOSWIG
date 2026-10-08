#include <ESP32Servo.h>
#include <ArduPID.h>

Servo steer_servo;
#define steer_servo_pin 14
const int max_left_steer_ms = 1226;        // max left in microseconds
const float max_left_steer_angle = -45;  // max left in angles
const int max_right_steer_ms = 1730;       // max right steer in microseconds
const float max_right_steer_angle = 45;   // max left in angles

Servo brake_servo;
#define brake_servo_pin 27         //brake servo
const int min_brake_servo = 1140;  //minimum braking (not braking at all)
const int max_brake_servo = 1300;  //maximum braking

#define speed_pin 25      // rover speed
#define rev_pin 4         // rover reverse
ArduPID speedController;

//DAC limits, for mapping speed control to a voltage
const int min_dac = 90;
const int max_dac = 200;
double setpoint_dac = 0;
int setpoint_brake = 50;

double p = 10;
double i = 1;
double d = 1;
unsigned long last_accel = micros();

//Variables to control rover (steering, speed, braking)
int steering_val = 0;
int speed_val = 0;
int braking_val = 0;

// To store control commands from the Jetson:
float steering_angle = -10;
float current_steering_angle = 10;
float steering_angle_velocity = 0;
double velocity = 0;
float acceleration = 0;
float jerk = 0;
double GPS_vel = 0;

// Keep track of time
long heartbeat_interval = 200000;  // 0.2 secondes
unsigned long time_last_sent_heartbeat = micros();
unsigned long time_last_received_heartbeat = micros();

// Declare variables to store previous values
float previous_steering_angle = 0;
double previous_setpoint_dac = 0;
int previous_setpoint_brake = 0;
const float tolerance = 0.01;


void setup() {
  Serial.begin(115200);

  steer_servo.attach(steer_servo_pin, max_left_steer_ms, max_right_steer_ms);
  steer_servo.writeMicroseconds((max_left_steer_ms + max_right_steer_ms) / 2);

  brake_servo.attach(brake_servo_pin, min_brake_servo, max_brake_servo);
  brake_servo.writeMicroseconds(map(setpoint_brake, 0, 100, min_brake_servo, max_brake_servo));

  pinMode(speed_pin, OUTPUT);
  pinMode(rev_pin, OUTPUT);
  digitalWrite(rev_pin, LOW);

  // To del ?
  // Slow start to avoid the rover speed controller going into safety lock
  while (setpoint_dac < min_dac) {
    dacWrite(speed_pin, setpoint_dac);
    setpoint_dac++;
    delay(10);
  }

  speedController.begin(&velocity, &setpoint_dac, &GPS_vel, p, i, d);
  speedController.setOutputLimits(-100, max_dac);
  // speedController.setOutputLimits(-100 + min_dac, max_dac);
  // speedController.setBias(min_dac);
  // speedController.setWindUpLimits(const double& min, const double& max);

  // Relay to disable 12V power when the ESP turns off, protects the buck converter
  pinMode(relay_pin, OUTPUT);
  digitalWrite(relay_pin, HIGH);

  Serial.println("Setup done!");
}


void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('>');  // Read until the end character
    input.remove(0, 1);                          // Remove the starting '<'

    // Split the string by commas and parse values
    parseSerialInput(input);

    // Update the last time we received data
    time_last_received_heartbeat = micros();
  }

  // Call the control functions
  runSteering();

  // Print only if the steering angle changed (within a tolerance for floats)
  if (abs(current_steering_angle - previous_steering_angle) > tolerance) {
    Serial.print("Steering changed: ");
    Serial.print("Current angle: ");
    Serial.print(current_steering_angle);
    Serial.print(" | Setpoint (us): ");
    Serial.println(steer_servo.readMicroseconds());

    // Update the previous steering angle
    previous_steering_angle = current_steering_angle;
  }

  runAcceleration();

  // Print only if the acceleration (DAC setpoint) or brake setpoint changed (with tolerance)
  if (abs(setpoint_dac - previous_setpoint_dac) > tolerance || abs(setpoint_brake - previous_setpoint_brake) > tolerance) {
    Serial.print("Acceleration changed: ");
    Serial.print("DAC setpoint: ");
    Serial.print(setpoint_dac);
    Serial.print(" | Brake setpoint: ");
    Serial.println(setpoint_brake);

    // Update the previous DAC and brake setpoints
    previous_setpoint_dac = setpoint_dac;
    previous_setpoint_brake = setpoint_brake;
  }

  // Send a heartbeat back to the Jetson to show we are alive
  // heartbeat();
}


void parseSerialInput(String input) {
  // <steering_angle,steering_angle_velocity,velocity,acceleration,jerk,GPS_velocity>
  // Split the input string by commas
  int commaIndex1 = input.indexOf(',');
  int commaIndex2 = input.indexOf(',', commaIndex1 + 1);
  int commaIndex3 = input.indexOf(',', commaIndex2 + 1);
  int commaIndex4 = input.indexOf(',', commaIndex3 + 1);
  int commaIndex5 = input.indexOf(',', commaIndex4 + 1);

  // Check if the expected number of commas is present
  if (commaIndex1 >= 0 && commaIndex2 >= 0 && commaIndex3 >= 0 && commaIndex4 >= 0) {
    // Parse the values safely
    steering_angle = input.substring(0, commaIndex1).toFloat();
    steering_angle_velocity = input.substring(commaIndex1 + 1, commaIndex2).toFloat();
    velocity = input.substring(commaIndex2 + 1, commaIndex3).toFloat();
    acceleration = input.substring(commaIndex3 + 1, commaIndex4).toFloat();
    jerk = input.substring(commaIndex4 + 1, commaIndex5).toFloat();
    GPS_vel = input.substring(commaIndex5 + 1).toFloat();

    // Process the parsed values as needed
    Serial.print("Received: ");
    Serial.print(steering_angle);           Serial.print(", ");
    Serial.print(steering_angle_velocity);  Serial.print(", ");
    Serial.print(velocity);                 Serial.print(", ");
    Serial.print(acceleration);             Serial.print(", ");
    Serial.print(jerk);                     Serial.print(", ");
    Serial.println(GPS_vel);
    
    Serial.print("Current angle: ");
    Serial.print(current_steering_angle);
    Serial.print(" | DAC Setpoint: ");
    Serial.println(setpoint_dac);
  }
  // else {
  //   Serial.println("Could not understand that message");
  //   Serial.println(current_steering_angle);
  //   Serial.println(steer_servo.read());
  // }
}

void runSteering() {
  // if (current_steering_angle != steering_angle) {
    float balance_steering = 0.001;
    float setAngle = (1-balance_steering) * current_steering_angle + balance_steering * steering_angle;
    int setPoint = int(map(setAngle, max_left_steer_angle, max_right_steer_angle, max_left_steer_ms, max_right_steer_ms));
    steer_servo.writeMicroseconds(setPoint);
    current_steering_angle = setAngle;
  // }
}

void runAcceleration() {
  float balance_acceleration = 0.1; // acceleration smoothing

  // Calculate the DAC value using empirical fomrula
  setpoint_dac = (1 - balance_acceleration) * setpoint_dac + balance_acceleration * (12.97 * abs(velocity) + 108.22);

  // Ensure setpoint_dac is within the allowed DAC limits
  setpoint_dac = constrain(setpoint_dac, min_dac, max_dac);

  // Determine the direction and braking
  if (velocity >= 1) {
    digitalWrite(rev_pin, LOW);  // Drive forward
    setpoint_brake = 0;
  }
  else if (velocity <= -1) {
    digitalWrite(rev_pin, HIGH); // Drive in reverse
    setpoint_brake = 0;
  }
  else {
    // Vehicle is near zero speed; apply brakes
    setpoint_dac = min_dac;
    setpoint_brake = 80;  // Adjust this value as needed for static braking intensity
  }

  // Write the DAC value to control speed
  dacWrite(speed_pin, setpoint_dac);

  // Control the brake servo
  brake_servo.writeMicroseconds(map(setpoint_brake, 0, 100, min_brake_servo, max_brake_servo));
}


void heartbeat() {
  // Let the Jetson know we are alive, check if we received a heartbeat back; otherwise stop the vehicle
  if (micros() - time_last_sent_heartbeat > heartbeat_interval) {
    Serial.print("Heartbeat: ");
    Serial.println(micros());
    time_last_sent_heartbeat = micros();
  }
  if (micros() - time_last_received_heartbeat > (heartbeat_interval * 1.5)) {
    // Set steering to center if no heartbeat for a while
    steering_angle = 0;  // Center the steering
    velocity = 0;
    // time_last_received_heartbeat = micros();
  }
}
