/*
 * Filename: axelBrake.ino
 *
 * Description:
 * This Arduino sketch implements a vehicle speed control system.
 * Target velocity and acceleration and actual messages are received via a MCP2515 CAN transiever.
 * Actual velocity and acceleration is based on GPS and IMU data from a CubePilot.
 * It controls acceleration and braking via PWM and servo signals.
 * The required signals are calculated use a PID controller.
 * The PWM acceleration signal is externally converted to a analog value to emulate the gas pedal.
 * The system also includes a heartbeat mechanism to maintain communication status, 
 * sending periodic heartbeat messages with the current control output.
 *
 * Features:
 * - Receives velocity and acceleration over CAN bus.
 * - Controls acceleration using PWM and braking using a servo.
 * - Implements a PID controller for velocity tracking.
 * - Sends periodic heartbeat messages with control output to monitor system status.
 * - Displays acceleration/braking setpoints on 7-segment displays.
 *
 * Usage:
 * Call the `heartbeat()` function in the main loop to send periodic heartbeats,
 * providing the necessary data to be sent as well as the CAN ID.
 * The system should be properly connected to the MCP2515 and the servo motor.
 *
 * CAN ID's:
 * 0x01: Heartbeat in from controller
 * 0x12: Heartbeat output / data logging
 * 0x120: Target Velocity message
 * 0x121: Current velocity message
 *
 * Dependencies:
 * - MCP2515 library for CAN communication.
 * - ArduPID library for PID control.
 * - Servo library for controlling the braking servo.
 *
 * Author: Matthijs Steyerberg, Axel Barbelanne
 * Date: 06-11-2024
 */

#include <SPI.h>
#include <mcp2515.h>
#include <ArduPID.h>
#include <Servo.h>


MCP2515 mcp2515(10);  // CAN CS to pin D10
Servo brakeServo;

// Define all the arduino pins used
const int canLED = 1;       // LED to display CAN messages NOT WORKING IF SERIAL ACTIVE
const int accelPin = 3;     // PWM output for acceleration
const int brakePin = 4;     // Servo control for braking
const int enablePin = 6;    // Relay to switch to autonomous acceleration overwrite
const int latchPin = 8;     // SH_CP for 7-segment display
const int clockPin = 7;     // ST_CP for 7-segment display
const int dataPin = 9;      // DS for 7-segment display

// Define range values for PWM and Servo (adjust as needed)
const int minBrake = 1900;  // Minimum pulse width for brake servo
const int maxBrake = 1200;  // Maximum pulse width for brake servo
const int minPWM = 38;      // Min PWM for acceleration (0% duty)
const int maxPWM = 225;     // Max PWM for acceleration (100% duty)

// PID values for velocity tracking
double P = 0.5; 
double I = 1e-4; 
double D = 1e-2;

ArduPID pidController;        // PID Controller
// PID targets
double targetVelocity = 0;    // Setpoint of control input
double actualGPSVelocity = 0; // Current GPS velocity
double controlOutput;         // Output of the PID controller
double targetAcceleration = 0; // Currently not used
double actualAcceleration = 0; // Currently not used

// Variables to hold the last displayed number and braking state
int lastDisplayedNumber = -1;
bool lastBrakingState = false;
byte standby = B00001000;     // Only show the dot
const unsigned long displayInterval = 700;
unsigned long lastDisplayTime = millis();

// Keep track of time
uint32_t heartbeatInterval = 200;   // ms
unsigned long nowMS = millis();
unsigned long lastHeartbeatReceived = nowMS;
unsigned long lastHeartbeatSent = 0;
bool receivedHeartbeat = false;

// CAN LED blinking if a message arrives
unsigned long lastCanSig = millis();
const unsigned long canLEDDur = 50;  // ms
bool can = false;

//---------------------  Setup  -------------------------------
int mode;   // 0 disarmed, 1 armed, 2 emergency

#define DEBUG_ENABLED     // Comment this line out to disable debug prints

#ifdef DEBUG_ENABLED
  #define Debug Serial
#else
  #define Debug if (false) Serial
#endif

void setup() {
  pinMode(canLED, OUTPUT);
  pinMode(accelPin, OUTPUT);
  brakeServo.attach(brakePin, minBrake, maxBrake);
  pinMode(enablePin, OUTPUT);
  pinMode(latchPin, OUTPUT);
  pinMode(clockPin, OUTPUT);
  pinMode(dataPin, OUTPUT);

  // Set default values
  digitalWrite(canLED, LOW);
  digitalWrite(enablePin, HIGH);   // disable overwrite
  analogWrite(accelPin, minPWM);       // no acceleration signal
  brakeServo.writeMicroseconds(minBrake);

  // Reset the display
  digitalWrite(latchPin, LOW);
  shiftOut(dataPin, clockPin, LSBFIRST, standby);
  shiftOut(dataPin, clockPin, LSBFIRST, standby);
  digitalWrite(latchPin, HIGH);

  Serial.begin(115200);  // Start Serial
  unsigned long startTime = millis();
  bool messageReceived = false;
  Serial.println(); Serial.println(); Serial.println();
  Serial.println("Send any message to keep serial connection alive");
  int countdown = 5;
  unsigned long timeOut = countdown * 1000;

  // Wait up to 5 seconds for a message from the Serial port
  while (millis() - startTime < timeOut) {
    if (Serial.available() > 0) {
      char receivedChar = Serial.read();  // Read the incoming byte
      Serial.print("Message received: ");
      Serial.println(receivedChar);
      messageReceived = true;  // Set the flag if any message is received
      break;
    }
    
    // Print countdown every second
    if (millis() - startTime >= (5000 - (countdown * 1000))) {
      Serial.print("Waiting for Serial input... ");
      Serial.print(countdown);
      Serial.println(" seconds remaining.");
      countdown--;
      delay(100);  // Small delay to sync with countdown
    }
    delay(10);  // Small delay to avoid excessive CPU usage
  }

  if (!messageReceived) {
    Serial.println("No message received. Ending Serial connection.");
    Serial.end();  // Stop Serial communication, freeing up pin 1
  } else {
    Serial.println("Serial connection remains active for debugging.");
  }
  
  SPI.begin();
  mcp2515.reset();
  mcp2515.setBitrate(CAN_500KBPS, MCP_8MHZ);
  mcp2515.setNormalMode();
  
  // Set PID parameters
  pidController.begin(&actualGPSVelocity, &controlOutput, &targetVelocity, P, I, D);
  pidController.setOutputLimits(-99, 99);  // Control range
  pidController.setWindUpLimits(-1, 1);
  pidController.start();
}


void loop() {
  struct can_frame canMsg;
  if (mcp2515.readMessage(&canMsg) == MCP2515::ERROR_OK) {
    Debug.println("Received CAN message:");
    // Check the CAN ID to differentiate between the messages
    if (canMsg.can_id == 0x01) {  // Heartbeat message
      Debug.println("\tMain Heartbeat");
      digitalWrite(canLED, HIGH); lastCanSig = millis();
      heartbeatInterval = (canMsg.data[0] << 8) | canMsg.data[1];
      mode = canMsg.data[2];
      lastHeartbeatReceived = millis();
      Debug.print("\t\tMode:\t\t");
      Debug.println(mode);
      Debug.print("\t\tInterval:\t");
      Debug.println(heartbeatInterval);
    }
    else if (canMsg.can_id == 0x120) {  // Target Velocity message
      digitalWrite(canLED, HIGH); 
      lastCanSig = millis();
      
      Debug.print("\ttarget velocity:\t");

      // Convert 4 bytes into a float (received CAN data is a float in 4 bytes)
      float receivedTargetVelocity;
      memcpy(&receivedTargetVelocity, canMsg.data, sizeof(receivedTargetVelocity));  // Copy 4 bytes from canMsg.data to receivedTargetVelocity
      targetVelocity = (double)receivedTargetVelocity;  // Cast the float to double

      // Convert 4 bytes into a float for acceleration
      float receivedTargetAcceleration;
      memcpy(&receivedTargetAcceleration, canMsg.data + 4, sizeof(receivedTargetAcceleration));  // Copy next 4 bytes for acceleration
      targetAcceleration = (double)receivedTargetAcceleration;  // Cast the float to double

      // Debugging output
      Debug.print(targetVelocity);
      Debug.print("\taccel:\t");
      Debug.println(targetAcceleration);
    }
    else if (canMsg.can_id == 0x121) {  // Actual Velocity message
      digitalWrite(canLED, HIGH); 
      lastCanSig = millis();
      
      Debug.print("\tactual velocity:\t");

      // Convert the first 4 bytes into a float for actual velocity
      float receivedActualVelocity;
      memcpy(&receivedActualVelocity, canMsg.data, sizeof(receivedActualVelocity));  // Copy first 4 bytes to receivedActualVelocity
      actualGPSVelocity = (double)receivedActualVelocity;  // Cast the float to double

      // Convert the next 4 bytes into a float for actual acceleration
      float receivedActualAcceleration;
      memcpy(&receivedActualAcceleration, canMsg.data + 4, sizeof(receivedActualAcceleration));  // Copy next 4 bytes for acceleration
      actualAcceleration = (double)receivedActualAcceleration;  // Cast the float to double

      // Debugging output
      Debug.print(actualGPSVelocity);
      Debug.print("\taccel:\t");
      Debug.println(actualAcceleration);
    }
    else {
      Debug.print("\tUndefined message id: ");
      Debug.println(canMsg.can_id);
    }
  }

  // Check emergency mode
  if (!receivedHeartbeat || mode == 2) {
    digitalWrite(enablePin, HIGH);
    analogWrite(accelPin, minPWM);
    digitalWrite(canLED, HIGH);
    brakeServo.writeMicroseconds(maxBrake);
  } 
  else if (mode == 0){    // disarmed mode (manual control)
    digitalWrite(enablePin, LOW);
    brakeServo.writeMicroseconds(minBrake);
  } 
  else {    // Autonomous control
    digitalWrite(enablePin, HIGH);
    
    if (targetVelocity <= 0) {    // Check if the target velocity is zero
      // Apply 50% braking
      int brakePosition = map(50, 0, 100, minBrake, maxBrake);
      brakeServo.writeMicroseconds(brakePosition);
      analogWrite(accelPin, minPWM); // Set acceleration to zero
      displayNumber(50, true); // Display 50 with braking dot
      controlOutput = -50;
    } 
    else{
      // Run the PID loop with the target and actual GPS velocities
      pidController.compute();
      Debug.print("\t\tComputed:\t\t");
      Debug.println(controlOutput);
      nowMS = millis();

      if (controlOutput >= 0) {
        // Positive output: control acceleration
        int pwmValue = map(controlOutput, 0, 100, minPWM, maxPWM);
        analogWrite(accelPin, pwmValue);
        brakeServo.writeMicroseconds(minBrake); // Release brake
        
        if (nowMS - lastDisplayTime > displayInterval){
          displayNumber(abs(controlOutput), false); // Show acceleration setpoint
          lastDisplayTime = millis();
        }
      } 
      else {
        // Negative output: control braking
        int brakePosition = map(-controlOutput, 0, 100, minBrake, maxBrake);
        brakeServo.writeMicroseconds(brakePosition);
        analogWrite(accelPin, minPWM); // Set acceleration to zero
        if (nowMS - lastDisplayTime > displayInterval){
          displayNumber(abs(controlOutput), true); // Show braking setpoint with dot
          lastDisplayTime = millis();
        }
      }
    }
  }
  
  byte data[8];
  float floatControlOutput = controlOutput;
  float floatActualVelocity = actualGPSVelocity;
  memcpy(data, &floatControlOutput, sizeof(floatControlOutput)); 
  memcpy(data + 4, &floatActualVelocity, sizeof(floatActualVelocity));
  receivedHeartbeat = heartbeat(0x12, data, sizeof(data));

  


  led_maintenance();
}


// Function to send a number to the 7-segment displays via 74HC595
void displayNumber(int number, bool braking) {
  // Only update the display if the number or braking state has changed
  if (number == lastDisplayedNumber && braking == lastBrakingState) {
    return; // No change, skip updating
  }
  // Debug.print("Displaying: ");   // Setpoint
  // Debug.print(number);
  // Debug.print("\tbraking: ");    // State: 0/1
  // Debug.print(braking);
  
  // Update the last displayed values
  lastDisplayedNumber = number;
  lastBrakingState = braking;

  byte digits[2] = { number / 10, number % 10 };  // Separate digits

  for (int i = 1; i >= 0; i--) {
    digitalWrite(latchPin, LOW); // Begin shifting in the data
    byte segmentValue = getSegmentValue(digits[i]);
    // Add dot to first display if braking
    if (braking) {
      segmentValue |= standby;  // Set dot on first display
    }
    
    Debug.print("\tdigit ");
    Debug.print(digits[i]);
    Debug.print("\tbyte: ");
    printByteAsBinary(segmentValue);

    shiftOut(dataPin, clockPin, LSBFIRST, segmentValue);
    digitalWrite(latchPin, HIGH); // Update display after both digits are shifted in
  }
  Debug.println();
}

void printByteAsBinary(byte value) {
  for (int i = 7; i >= 0; i--) {
    Debug.print(bitRead(value, i));
  }
}

// Map digit to 7-segment display pattern
byte getSegmentValue(byte digit) {
  switch (digit) {
    case 0: return 0b01110111;  // Segments a, b, c, d, e, f (no g)
    case 1: return 0b00010100;  // Segments b, c
    case 2: return 0b10110011;  // Segments a, b, d, e, g
    case 3: return 0b10110110;  // Segments a, b, c, d, g
    case 4: return 0b11010100;  // Segments b, c, f, g
    case 5: return 0b11100110;  // Segments a, c, d, f, g
    case 6: return 0b11100111;  // Segments a, c, d, e, f, g
    case 7: return 0b00110100;  // Segments a, b, c
    case 8: return 0b11110111;  // All segments
    case 9: return 0b11110110;  // Segments a, b, c, d, f, g
    default: return 0b00000000; // Turn off all segments
  }
}

// Function to manage heartbeat sending and receiving
bool heartbeat(uint32_t id, const byte* data, size_t dataSize) {
  unsigned long currentTime = millis();

  // Check that dataSize does not exceed 8 bytes
  if (dataSize > 8) {
    Debug.println("Invalid heartbeat data size");
    return false; // Invalid data size
  }

  // Send heartbeat if the interval has passed
  if (currentTime - lastHeartbeatSent >= heartbeatInterval) {
    sendHeartbeat(id, data, dataSize);  // Send heartbeat with provided data
    lastHeartbeatSent = currentTime;
  }

  // Check if the last received heartbeat is within the acceptable time window
  if (currentTime - lastHeartbeatReceived >= heartbeatInterval * 2) {
    // Heartbeat timeout logic (e.g., stop vehicle, apply brakes)
    return false; // Heartbeat not received in time
  }

  return true; // Heartbeat has been received recently
}


void sendHeartbeat(uint32_t id, const byte* data, size_t dataSize) {
  Debug.println("Sending Heartbeat Message:");
  
  // Debug output for CAN ID and data length
  Debug.print("\tCAN ID:\t\t0x");
  Debug.println(id, HEX);
  Debug.print("\tData Length:\t");
  Debug.println(dataSize);

  // Debug output for data bytes
  Debug.print("\tData Bytes:\t");
  for (size_t i = 0; i < dataSize; i++) {
    Debug.print("0x");
    Debug.print(data[i], HEX);
    Debug.print(i < dataSize - 1 ? ", " : "\n");
  }

  struct can_frame heartbeatMsg;
  heartbeatMsg.can_id = id;          // Set the CAN ID
  heartbeatMsg.can_dlc = dataSize;   // Set the data length
  memcpy(heartbeatMsg.data, data, dataSize); // Copy data to CAN message

  mcp2515.sendMessage(&heartbeatMsg); // Send the message
}


void led_maintenance(){
  nowMS = millis();

  if (nowMS - lastCanSig > canLEDDur){
    digitalWrite(canLED, LOW);
  }
}