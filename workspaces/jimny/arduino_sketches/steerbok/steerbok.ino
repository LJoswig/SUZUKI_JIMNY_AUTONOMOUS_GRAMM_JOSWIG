/*
 * Filename: steerbok.ino
 *
 * Description:
 * This Arduino sketch implements a vehicle steering control system.
 * Target steering angle messages are received via an MCP2515 CAN transceiver.
 * The current steering angle is read from a potentiometer, while a stepper motor 
 * is used to adjust the steering based on the desired angle.
 * A PID controller calculates the required steps for the stepper motor 
 * to reach the target angle smoothly.
 * The system includes a heartbeat mechanism to maintain communication status, 
 * sending periodic heartbeat messages with the current step output.
 *
 * Features:
 * - Receives target steering angle over CAN bus.
 * - Controls steering using a stepper motor based on PID control.
 * - Sends periodic heartbeat messages to monitor system status.
 * - Displays current and desired steering angles in the debug output.
 *
 * CAN IDs:
 * 0x01: Heartbeat in from controller
 * 0x11: Heartbeat output / data logging
 * 0x110: Target steering angle message
 *
 * Dependencies:
 * - MCP2515 library for CAN communication.
 * - ArduPID library for PID control.
 * - AccelStepper library for stepper interface
 *
 * Author: Axel Barbelanne, Matthijs Steyerberg
 * Date: 28-10-2024
 */

#include <EnableInterrupt.h>
#include <SPI.h>
#include <mcp2515.h>
#include <AccelStepper.h>
#include "ArduPID.h"

//---------------------  Parameters  -------------------------------

// Pins
const int pulPin = 5;       // Pulse pin to turn
const int dirPin = 4;       // Direction pin for steering
const int enaPin = 3;       // Enable pin to turn on/off steering
const int clkPin = A3;      // Encoder CLK pin
const int dtPin = A1;       // Encoder DT pin

const int pulRightLedPin = 6;
const int pulLeftLedPin = 7;
const int canLedPin = 8;

// PID
double kp = 12.0, ki = 0.5, kd = 0.5;

// CAN
const int canHeartbeatInId = 0x01;
const int canLoggingId = 0x11;
const int canDesAngleId = 0x110;

// Encoder
volatile long counter = 0;        // Total encoder count
volatile uint8_t lastState = 0;   // Last state of the encoder (for transition tracking)
const int encoderPPR = 4000;      // Pulses per revolution
const double sprocketGearRatio = 2.5; // Gear ratio
const bool encoderCwRight = false; // Clockwise is right turn

// Stepper
const int maxStepperSpeed = 800;    // Pulses per second
const bool stepCwRight = true;     // Clockwise rotation of stepper corresponds to right turn
bool stepperOn = false;             // For stepper LEDs

// Keep track of time
uint32_t heartbeatInterval = 200;   // ms: 0.2s -> 5Hz
unsigned long nowMs = millis();
unsigned long lastHeartbeatReceived = nowMs;
unsigned long lastHeartbeatSent = nowMs;

// LED blinking
unsigned long lastCanSig = nowMs;
unsigned long lastPulRightSig = nowMs;
unsigned long lastPulLeftSig = nowMs;
const unsigned long canLedDur = 50;    // ms
unsigned long pulLedDur = 250;          //ms

//---------------------  Setup  -------------------------------

int mode;   // 0 disarmed, 1 armed, 2 emergency

// CAN
struct can_frame canMsg;
MCP2515 mcp2515(10);

// PID
double currAngle, outputRate, desAngle; 
ArduPID pidController;

// Stepper
AccelStepper stepper(AccelStepper::DRIVER, pulPin, dirPin);

// #define DEBUG_ENABLED     // Comment this line out to disable debug prints

#ifdef DEBUG_ENABLED
  #define Debug Serial
#else
  #define Debug if (false) Serial
#endif

//---------------------  CODE  -------------------------------

void handleEncoder(); // Interrupt handler

// Setup function
void setup() {
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

  // Angular encoder
  pinMode(clkPin, INPUT_PULLUP);
  pinMode(dtPin, INPUT_PULLUP);
  lastState = (digitalRead(clkPin) << 1) | digitalRead(dtPin);  // Encoder state
  enableInterrupt(clkPin, handleEncoder, CHANGE);
  enableInterrupt(dtPin, handleEncoder, CHANGE);


  // LED's
  pinMode(pulRightLedPin, OUTPUT);
  pinMode(pulLeftLedPin, OUTPUT);
  pinMode(canLedPin, OUTPUT);
  digitalWrite(pulRightLedPin, LOW);
  digitalWrite(pulLeftLedPin, LOW);
  digitalWrite(canLedPin, LOW);

  // MCP setup for CAN
  mcp2515.reset();
  mcp2515.setBitrate(CAN_500KBPS, MCP_8MHZ);
  mcp2515.setNormalMode();
  
  // Initialize PID controller
  pidController.begin(&currAngle, &outputRate, &desAngle, kp, ki, kd);
  pidController.setOutputLimits(-maxStepperSpeed, maxStepperSpeed);
  pidController.setWindUpLimits(-10, 10);
  if(stepCwRight)
    pidController.reverse();
  pidController.start();

  // Initialize Stepper
  stepper.setEnablePin(enaPin);
  stepper.setPinsInverted(false, false, true);
  stepper.disableOutputs();
  stepper.setMaxSpeed(maxStepperSpeed);
  stepper.setSpeed(0);
}

void loop() {
  nowMs = millis();
  // Constantly sCAN
  if (mcp2515.readMessage(&canMsg) == MCP2515::ERROR_OK) {
    Debug.println("Received CAN Message:");
    if (canMsg.can_id == canHeartbeatInId) {
      Debug.println("\tMain Heartbeat");
      digitalWrite(canLedPin, HIGH); lastCanSig = millis();
      heartbeatInterval = (canMsg.data[0] << 8) | canMsg.data[1];
      mode = canMsg.data[2];
      lastHeartbeatReceived = millis();
      Debug.print("\t\tMode:\t\t");
      Debug.println(mode);
      Debug.print("\t\tInterval:\t");
      Debug.println(heartbeatInterval);
    } else if (canMsg.can_id == canDesAngleId) {
      Debug.print("\ttarget steer angle:\t");
      digitalWrite(canLedPin, HIGH);
      lastCanSig = nowMs;
      
      float receivedAngle;
      memcpy(&receivedAngle, canMsg.data, sizeof(receivedAngle));  // Copy 4 bytes from canMsg.data to receivedAngle
      desAngle = (double)receivedAngle; // Convert from float to double
      Debug.println(desAngle);
      stepper.enableOutputs();
      stepperOn = true;
    } else {
      Debug.print("\tUndefined message id: ");
      Debug.println(canMsg.can_id);
    }
  }

  currAngle = getCurrAngle();

  pidController.compute();
  // Update new speed for stepper
  stepper.setSpeed(outputRate);
  // Shine right or left LED
  shineLed(outputRate);

  // Logging and heartbeat
  byte data[8];  // 8 bytes to hold two floats (4 bytes each)

  float floatOutput = outputRate;
  float floatCurrAngle = currAngle;

  // Log PID output
  memcpy(data, &floatOutput, sizeof(floatOutput));

  // Log current angle (feedback)
  memcpy(data + 4, &floatCurrAngle, sizeof(floatCurrAngle));

  bool receivedHeartbeat = heartbeat(canLoggingId, data, sizeof(data));

  if (!receivedHeartbeat) {
    // DEBUG:
    stepper.disableOutputs();
    stepperOn = false;
    Debug.println("No heartbeat received, shutting down");
    receivedHeartbeat = false;
  }
  

  Debug.print("Desired Angle: "); Serial.print(desAngle);
  Debug.print(" | Current Angle: "); Serial.print(currAngle);
  Debug.print(" | PID Output Step: "); Serial.println(outputRate);

  // Switch off LEDs
  ledMaintenance();

  // Continuously run stepper
  stepper.runSpeed();
}


// Interrupt handler for angular encoder
void handleEncoder() {
  // Directly read PORTC for A0 (PC0) and A1 (PC1)
  uint8_t clkState = (PINC & (1 << PC3)) ? 1 : 0;
  uint8_t dtState = (PINC & (1 << PC1)) ? 1 : 0;

  // Combine states into a single byte
  uint8_t newState = (clkState << 1) | dtState;

  // Use state transition table for reliable counting
  switch ((lastState << 2) | newState) {
    case 0b0001: case 0b0111: case 0b1110: case 0b1000: // CW transitions
      counter++;
      break;
    case 0b0010: case 0b1011: case 0b1101: case 0b0100: // CCW transitions
      counter--;
      break;
    default:
      break; // Ignore invalid transitions
  }

  lastState = newState;
}

// Function to read the current steering angle from encoder
double getCurrAngle() {
  // Convert encoder counter to angle
  double rotations = double(counter) / encoderPPR;
  double angle = 360.0 * rotations / sprocketGearRatio;
  Debug.println("Angular encoder"); Debug.println(counter);
  angle *= encoderCwRight ? -1 : 1; // Adjust for clockwise direction
  return angle;
}

// Function to manage heartbeat sending and receiving
bool heartbeat(uint32_t id, const void* data, size_t dataSize) {
  unsigned long currentTime = millis();
  
  if (dataSize > 8) {
    Serial.println("Invalid heartbeat data size");
    return false; // Invalid data size
  }

  if (currentTime - lastHeartbeatSent >= heartbeatInterval) {
    sendHeartbeat(id, data, dataSize);
    lastHeartbeatSent = currentTime;
  }

  if (currentTime - lastHeartbeatReceived >= heartbeatInterval * 2) {
    return false; // Heartbeat not received in time
  }

  return true; // Heartbeat has been received recently
}

// Function to manage heartbeat sending
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
  heartbeatMsg.can_id = id;
  heartbeatMsg.can_dlc = dataSize;
  memcpy(heartbeatMsg.data, data, dataSize);
  mcp2515.sendMessage(&heartbeatMsg);
}

// Function to manage turning on the left/right LED's
void shineLed(int outputRate){
  pulLedDur = map(abs(outputRate), 0, maxStepperSpeed, 250, 25);
  nowMs = millis();
  // Don't shine anything if stepper is off
  if (!stepperOn)
    return;
  // If turning right
  if (outputRate > 0 ^ stepCwRight){
    if((nowMs - lastPulRightSig > 2 * pulLedDur)){
      digitalWrite(pulRightLedPin, HIGH);
      lastPulRightSig = nowMs;
    }
  }
    // If turning left
  else if((nowMs - lastPulLeftSig > 2 * pulLedDur)){
    digitalWrite(pulLeftLedPin, HIGH);
    lastPulLeftSig = nowMs;
  }
}

// Function to manage turning off the LED's for appropriate durations
void ledMaintenance(){
  nowMs = millis();

  if (nowMs - lastCanSig > canLedDur){
    digitalWrite(canLedPin, LOW);
  }
  if (nowMs - lastPulRightSig > pulLedDur){
    digitalWrite(pulRightLedPin, LOW);
  }
  if (nowMs - lastPulLeftSig > pulLedDur){
    digitalWrite(pulLeftLedPin, LOW);
  }
}