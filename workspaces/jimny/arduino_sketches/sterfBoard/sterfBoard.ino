/*
 * Filename: sterfBoard.ino
 *
 * Description:
 * This Arduino sketch implements a command pass through and power switching for all submodules.
 * Includes emergency stop button, if pressed the module will stop the vehicle and disable the steering stepper.
 *
 * Features:
 * - Emergency stop
 * - Mode select buttons
 * - Send/Receive commands/measurements over Serial Connection
 * - Send/Receive commands/measurements over CAN bus
 *
 * Usage:
 * Connect the arduino to the Jetson to receive commands and states.
 * The arduino will send back the states received from all modules.
 *
 * Dependencies:
 * - MCP2515 library for CAN communication.
 * - ArduPID library for PID control.
 *
 * Author: Matthijs Steyerberg, Axel Barbelanne
 * Date: 06-11-2024
 */

#include <SPI.h>
#include <mcp2515.h>

MCP2515 mcp2515(10);  // CAN CS to pin D10

// Define all the arduino pins used
const int relay48v = 3;
const int relay9v = 4;
const int eButton = 5;
const int autoButton = 6;
const int res1Button = 7;
const int res2Button = 8;
const int canLED = 9;
unsigned long canLEDtime = 0;
const int cubePWM = A0;

// Heartbeat configuration
const uint32_t heartbeatId = 0x01;
const uint32_t heartbeatInterval = 200; // 5 Hz
unsigned long nowMS = millis();
unsigned long lastHeartbeatTime = 0;
unsigned long lastSerialHeartbeatTime = nowMS;
bool serialHeartbeatReceived = false;

// boolean declaring error in ackermann signal
bool ackSignalInvalid = true;
unsigned long lastAckTime = nowMS;
unsigned long noAckTimeout = 500; // 500ms
bool noAckReceived = true;
int hbSignal = 0;

// CAN message IDs
// Receiving Heartbeats
const uint32_t steerBok = 0x11;
const uint32_t axelBrake = 0x12;
// Sending data
const uint32_t steeringId = 0x110;
const uint32_t velocityId = 0x120;
const uint32_t gpsVelocityId = 0x121;

// CAN LED blinking if a message arrives
unsigned long lastCanSig = nowMS;
const unsigned long canLEDDur = 50;  // ms

// Parsed data variables
float steeringAngle, steeringAngleVelocity, velocity, acceleration, jerk, gpsVelocity;

// Received data variables from heartbeats
float velocitySetpoint = 0;
float velocityFeedback = 0;
float steeringRateSetPoint = 0;
float steeringAngleFeedback = 0;

// Signal received flag from Jetson
enum HbSignal {
    DISARMED = 0,
    VALID_ARMED = 1,
    INVALID_ARMED = 2
};

// mode, to be published in CAN to other modules
enum Mode {
    MANUAL = 0,
    AUTO = 1,
    EMERGENCY = 2
};

// #define DEBUG_ENABLED     // Comment this line out to disable debug prints

#ifdef DEBUG_ENABLED
  #define Debug Serial
#else
  #define Debug if (false) Serial
#endif


void setup() {
  Serial.begin(115200);
  Debug.println(); Debug.println(); Debug.println();

  SPI.begin();
  mcp2515.reset();
  mcp2515.setBitrate(CAN_500KBPS, MCP_8MHZ);
  mcp2515.setNormalMode();

  Debug.println("CAN Serial Bridge Initialized.");

  pinMode(relay48v, OUTPUT);
  pinMode(relay9v, OUTPUT);
  pinMode(eButton, INPUT);
  pinMode(autoButton, INPUT_PULLUP);    // Internal PULLUP resistors do not seem to work?
  pinMode(res1Button, INPUT_PULLUP);
  pinMode(res2Button, INPUT_PULLUP);
  pinMode(canLED, OUTPUT);

  digitalWrite(relay9v, LOW);
  digitalWrite(relay48v, HIGH);
  Debug.println("Input/Output pins Initialized.");
}


void loop() {
  // Check for serial data input and parse if available
  unsigned long now = millis();
  if (digitalRead(eButton) == LOW) {
    Debug.println("Emergency mode!");
    if (digitalRead(relay48v) == HIGH) {
      sendHeartbeatMessage(EMERGENCY);
      lastHeartbeatTime = now;
      digitalWrite(relay48v, LOW);
    }
    else if (now - lastHeartbeatTime >= heartbeatInterval){
      sendHeartbeatMessage(EMERGENCY);
      lastHeartbeatTime = now;
    }
  }
  else if (digitalRead(autoButton) == LOW) {
    if (digitalRead(relay48v) == HIGH) {
      sendHeartbeatMessage(MANUAL);
      lastHeartbeatTime = now;
      digitalWrite(relay48v, LOW);
    }
    else if (now - lastHeartbeatTime >= heartbeatInterval){
      sendHeartbeatMessage(MANUAL);
      lastHeartbeatTime = now;
    }
  }
  else {

    // ---------- Serial communication with the Jetson ----------
    if (Serial.available() > 0) {
      Debug.println("Received Serial message:");
      String inputData = Serial.readStringUntil('>');  // Read until the end character
      inputData.remove(0, 1);                          // Remove the starting '<'
      if (inputData.startsWith("H")){
        hbSignal = inputData.substring(2).toInt();
        if (hbSignal == 0){
          Debug.println("Disarmed");
        }
        else if (hbSignal == 2){
          Debug.println("Ackermann signal invalid");
        }
        lastSerialHeartbeatTime = now;
        serialHeartbeatReceived = true;
      }
      else{
        parseSerialData(inputData);
      }
    }
    if (now - lastSerialHeartbeatTime >= 2 * heartbeatInterval) {
      // Debug.println("No serial heartbeat received!");
      serialHeartbeatReceived = false;
    }

    // ---------- CAN bus communication ----------
    // Send heartbeat message at the specified interval
    now = millis();
    if (now - lastHeartbeatTime >= heartbeatInterval) {
      //check if no ackermann received when it was supposed to be received
      if ((hbSignal == VALID_ARMED) && (now - lastAckTime > noAckTimeout)){
        noAckReceived = true;
        Debug.println("Expected Ackermann message but did not receive one.");
      }
      else{
        noAckReceived = false;
      }

      int checkedMode;
      switch (hbSignal){
        case DISARMED: 
          checkedMode = MANUAL;
          break;
        case VALID_ARMED: 
          checkedMode = AUTO;
          break;
        case INVALID_ARMED: 
          checkedMode = EMERGENCY;
          break;
        default:
          Debug.println("Error: Incorrect heartbeat flag received from heartbeat message");
          break;
      }

      Debug.print("Checked Mode: "); Debug.println(checkedMode);

      if (checkedMode == AUTO){
        digitalWrite(relay48v, HIGH);
      }
      else{
        digitalWrite(relay48v, LOW);
      }
      sendHeartbeatMessage(checkedMode);
      lastHeartbeatTime = now;
      
      // Send CAN messages for velocity, acceleration, GPS velocity, and steering angle/rate
      sendCANMessages();
    }

    // Check for incoming CAN messages and process ID 0x10 for control setpoints
    checkCANMessages();
    
    // CAN LED blinking
    led_maintenance();
  }
}

// Function to parse incoming serial data
void parseSerialData(String inputData) {
  float values[6];
  int index = 0;
  char* token = strtok(inputData.c_str(), ",");

  // Tokenize the string and parse values
  while (token != NULL && index < 6) {
    values[index++] = atof(token);
    token = strtok(NULL, ",");
  }

  // Assign parsed values to variables if all 6 values were received
  if (index == 6) {
    lastAckTime = millis();
    steeringAngle = values[0];
    steeringAngleVelocity = values[1];
    velocity = values[2];
    acceleration = values[3];
    jerk = values[4];
    gpsVelocity = values[5];

    // Debug print statements for each parsed value
    Debug.println("Parsed Serial Data:");
    Debug.print("\tSteering angle:\t"); Debug.println(steeringAngle);
    Debug.print("\tSteering rate:\t"); Debug.println(steeringAngleVelocity);
    Debug.print("\tVelocity:\t"); Debug.println(velocity);
    Debug.print("\tAcceleration:\t"); Debug.println(acceleration);
    Debug.print("\tJerk:\t\t"); Debug.println(jerk);
    Debug.print("\tGPS Velocity:\t"); Debug.println(gpsVelocity);
  } else {
    // Error if data was incomplete
    Debug.println("Error: Incomplete data received.");
  }
}

// Send a heartbeat message with mode and interval
void sendHeartbeatMessage(int mode) {
  Debug.println("Sending heartbeat: ");
  struct can_frame heartbeatMsg;
  heartbeatMsg.can_id = heartbeatId;
  heartbeatMsg.can_dlc = 4;

  heartbeatMsg.data[0] = heartbeatInterval >> 8;  // High byte
  heartbeatMsg.data[1] = heartbeatInterval & 0xFF; // Low byte
    // Emergency mode
  if (mode == EMERGENCY) {
    heartbeatMsg.data[2] = 0x02; 
  } 
  // Manual mode
  else if (mode == MANUAL){ 
    heartbeatMsg.data[2] = 0x00;
  }
  // Autonomous mode
  else if (mode == AUTO){
    heartbeatMsg.data[2] = 0x01; // 1:autonomous if data received and valid, 0:manual otherwise)
  }
  else{
    Debug.println("Unsupported mode: use 0, 1 or 2");
    return;
  }
  heartbeatMsg.data[3] = 0x00; // necessary to fill can message

  mcp2515.sendMessage(&heartbeatMsg);

  // Output the complete data for debugging
  Debug.print("\tCAN Heartbeat:\t");
  for (int i = 0; i < heartbeatMsg.can_dlc; i++) {
    Debug.print("0x");
    Debug.print(heartbeatMsg.data[i], HEX);
    if (i < heartbeatMsg.can_dlc - 1) Debug.print(", ");
  }
  Debug.println();

  // Send the control setpoint and module data back to the serial port
  Debug.print("\tSerial Heartbeat:\t");
  Serial.print("<");
  Serial.print(velocitySetpoint);       Serial.print(", ");
  Serial.print(velocityFeedback);       Serial.print(", ");
  Serial.print(steeringRateSetPoint);   Serial.print(", ");
  Serial.print(steeringAngleFeedback);
  Serial.println(">");
}

// Function to send CAN messages with parsed data
void sendCANMessages() {
  Debug.println("Sending CAN messages:");
  // Steering angle and rate
  struct can_frame steeringMsg;
  steeringMsg.can_id = steeringId;
  steeringMsg.can_dlc = 8;
  floatToBytes(steeringAngle, steeringMsg.data);              // Steering angle (4 bytes)
  floatToBytes(steeringAngleVelocity, &steeringMsg.data[4]); // Steering angle velocity (4 bytes)
  mcp2515.sendMessage(&steeringMsg);

  // Debug output for steering data
  Debug.println("\tSteerBok:");
  Debug.print("\t\tAngle:\t");
  Debug.println(steeringAngle);
  Debug.print("\t\tRate:\t");
  Debug.println(steeringAngleVelocity);
  Debug.print("\t\tData:\t");
  for (int i = 0; i < steeringMsg.can_dlc; i++) {
    Debug.print("0x");
    Debug.print(steeringMsg.data[i], HEX);
    Debug.print(i < steeringMsg.can_dlc - 1 ? ", " : "\n");
  }

  // Velocity and acceleration
  struct can_frame velocityMsg;
  velocityMsg.can_id = velocityId;
  velocityMsg.can_dlc = 8;
  floatToBytes(velocity, velocityMsg.data);              // Velocity (4 bytes)
  floatToBytes(acceleration, &velocityMsg.data[4]);      // Acceleration (4 bytes)
  mcp2515.sendMessage(&velocityMsg);

  // Debug output for velocity and acceleration data
  Debug.println("\tAxelBrake:");
  Debug.print("\t\tVel:\t");
  Debug.println(velocity);
  Debug.print("\t\tAcc:\t");
  Debug.println(acceleration);
  Debug.print("\t\tData:\t");
  for (int i = 0; i < velocityMsg.can_dlc; i++) {
    Debug.print("0x");
    Debug.print(velocityMsg.data[i], HEX);
    Debug.print(i < velocityMsg.can_dlc - 1 ? ", " : "\n");
  }

  // GPS velocity
  struct can_frame gpsVelocityMsg;
  gpsVelocityMsg.can_id = gpsVelocityId;
  gpsVelocityMsg.can_dlc = 4;
  floatToBytes(gpsVelocity, gpsVelocityMsg.data);
  mcp2515.sendMessage(&gpsVelocityMsg);

  Debug.print("\t\tGPS Vel:");
  Debug.println(gpsVelocity);
  Debug.print("\t\tData:\t");
  for (int i = 0; i < gpsVelocityMsg.can_dlc; i++) {
    Debug.print("0x");
    Debug.print(gpsVelocityMsg.data[i], HEX);
    Debug.print(i < gpsVelocityMsg.can_dlc - 1 ? ", " : "\n");
  }
}

// Function to convert float to 4 bytes
void floatToBytes(float value, uint8_t* bytes) {
  union {
    float f;
    uint8_t b[4];
  } converter;
  converter.f = value;
  bytes[0] = converter.b[0];
  bytes[1] = converter.b[1];
  bytes[2] = converter.b[2];
  bytes[3] = converter.b[3];
}

// Check for incoming CAN messages
void checkCANMessages() {
  struct can_frame receivedMsg;
  if (mcp2515.readMessage(&receivedMsg) == MCP2515::ERROR_OK) {
    Debug.println("Received CAN message: ");
    if (receivedMsg.can_dlc != 8) {
      Debug.println("\tError: CAN message is not 8 bytes. Ignoring message.");
      return; // Skip processing this message
    }
    if (receivedMsg.can_id == axelBrake) {
      Debug.print("\taxelBrake: ");
      digitalWrite(canLED, HIGH); lastCanSig = millis();
      memcpy(&velocitySetpoint, &receivedMsg.data[0], sizeof(velocitySetpoint));

      memcpy(&velocityFeedback, &receivedMsg.data[4], sizeof(velocityFeedback));

      Debug.print("velocitySetpoint: ");
      Debug.println(velocitySetpoint);
      
      Debug.print("velocityFeedback: ");
      Debug.println(velocityFeedback);
    }
    else if (receivedMsg.can_id == steerBok) {
      Debug.print("\tsteerBok: ");
      digitalWrite(canLED, HIGH); lastCanSig = millis();
      // Extract the first 4 bytes for steeringRateSetPoint
      memcpy(&steeringRateSetPoint, &receivedMsg.data[0], sizeof(steeringRateSetPoint));
      
      // Extract the next 4 bytes for steeringAngleFeedback
      memcpy(&steeringAngleFeedback, &receivedMsg.data[4], sizeof(steeringAngleFeedback));
      
      Debug.print("steeringSetPoint: ");
      Debug.println(steeringRateSetPoint);
      
      Debug.print("steeringAngleFeedback: ");
      Debug.println(steeringAngleFeedback);
    }
  }
}

void led_maintenance(){
  nowMS = millis();

  if (nowMS - lastCanSig > canLEDDur){
    digitalWrite(canLED, LOW);
  }
}