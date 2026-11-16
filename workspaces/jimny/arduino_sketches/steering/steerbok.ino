/*
 * steering_can.ino
 *
 * Versione modificata:
 * encoder su D4 e D5 invece di A3(verde) e A1(bianco)
 * serialDirectMode aggiunto per test diretto PC -> steering board
 */

#include <Wire.h>
#include <Adafruit_MCP4728.h>
#include <EnableInterrupt.h>
#include <SPI.h>
#include <mcp2515.h>
#include <avr/wdt.h>     
#include "can_ids.h"

#define STEERING_USE_CAN 0

// =========================
// ENCODER
// =========================
const int clkPin = 4;   // D4 = CHA
const int dtPin  = 5;   // D5 = CHB

volatile long counter = 0;
volatile uint8_t lastState = 0;
volatile unsigned long isrEvents = 0;
volatile unsigned long invalidTransitions = 0;

int encoderCountsPerRev = 4000;
double sprocketGearRatio = 2.5;
bool encoderCwRight = false;

// =========================
// DAC
// =========================
Adafruit_MCP4728 mcp;

const float DAC_VDD     = 4.98;
const int   DAC_MAX     = 4095;
const float IDLE_A      = 2.5200;
const float IDLE_B      = 2.4900;
const float DELTA_LIMIT = 0.80;
const float I_ZONE_DEG = 5.0;   // NEW: integrate only when |error| < 5 deg

float CAL_A = 0.0000;
float CAL_B = 0.0000;

float    lastOutA  = IDLE_A;
float    lastOutB  = IDLE_B;
uint16_t lastCodeA = 0;
uint16_t lastCodeB = 0;

// =========================
// PID
// =========================
float targetAngleDeg   = 0.0;
float measuredAngleDeg = 0.0;

// --- Slew rate limiter globals ---
float targetSlewed = 0.0;
const float MAX_TARGET_RATE = 30.0; // deg/s max ramp speed
// ---------------------------------

float Kp = 0.020;
float Ki = 0.012;
float Kd = 0.0012;

float integralTerm  = 0.0;
const float integralLimit = 40.0;

float prevAngleDeg = 0.0;
float deltaCmd     = 0.0;

float lastErrorDeg = 0.0;
float lastPTerm    = 0.0;
float lastITerm    = 0.0;
float lastDTerm    = 0.0;

bool pidEnabled = false;
bool serialDirectMode = false;

// Latched hold, only meaningful while pidEnabled. Set by "x" (emergency from the Jetson)
// or by the serial timeout. While set, s<deg> is ignored. Cleared only by "e" or "d".
// State reported in the DATA line: 0 = off, 1 = tracking, 2 = holding.
bool holdLatched = false;

// --- CHANGED: error deadband now has hysteresis (replaces ERROR_DEADBAND_DEG) ---
// Enter the deadband when |error| < DB_ENTER_DEG, leave it only when |error| > DB_EXIT_DEG.
// Inside the deadband: no P action, integral is FROZEN (kept, not reset) -> no output step.
// Set DB_ENTER_DEG slightly above half the measured peak-to-peak swing.
// DB_EXIT_DEG is the largest error you accept without correcting.
// 0.0 / 0.0 = deadband disabled (same behaviour as before).
const float DB_ENTER_DEG = 0.1;   // NEW: e.g. 0.4
const float DB_EXIT_DEG  = 0.0;   // NEW: e.g. 0.8 (must be >= DB_ENTER_DEG)
bool inDeadband = false;          // NEW
// Output deadband kept, but best left at 0 (it fights the integrator)
const float OUTPUT_DEADBAND_V  = 0.00;
// ----------------------------------------------

// =========================
// TIMING
// =========================
const unsigned long controlPeriodMs = 10;
unsigned long lastControlMs = 0;

const unsigned long printIntervalMs = 50;
unsigned long lastPrintMs = 0;

// =========================
// NEW: SERIAL WATCHDOG
// =========================
// If PID is on in serial direct mode and no serial line (any command or the "h" keepalive)
// arrives for SERIAL_TIMEOUT_MS, the target is set to the current angle -> steering holds there.
// [ESTOP] The hold is latched: serial lines arriving again do NOT resume tracking, only "e" does.
// [ESTOP] The Jetson bridge sends "h" every 100 ms. For manual typing tests send "h" or raise the timeout.
// [ESTOP] removed: When serial lines arrive again, normal control resumes (target stays at the held angle
// [ESTOP] removed: until a new s<deg> command).
unsigned long lastSerialRxMs = 0;                // NEW
const unsigned long SERIAL_TIMEOUT_MS = 1000;     // [ESTOP] was: 60000  //500    // NEW: raise this for manual typing tests
// [ESTOP] removed: bool serialLost = false;   (replaced by holdLatched)
// =========================
// CAN
// =========================
MCP2515 mcp2515(10);
struct can_frame canMsg;

int      mode                = 0;
uint32_t heartbeatInterval   = 200;
unsigned long lastHbReceived = 0;
unsigned long lastHbSent     = 0;

unsigned long lastCanSig       = 0;
const unsigned long CAN_LED_DUR = 50;

// #define DEBUG_ENABLED
#ifdef DEBUG_ENABLED
  #define Debug Serial
#else
  #define Debug if(false) Serial
#endif

// =========================
// HELPERS
// =========================
double getSteeringAngleDeg(long countsSnapshot) {
  double revs  = (double)countsSnapshot / (double)encoderCountsPerRev;
  double angle = 360.0 * revs / sprocketGearRatio;
  return angle * (encoderCwRight ? -1.0 : 1.0);
}

uint16_t voltToCode(float v) {
  if (v < 0.0) v = 0.0;
  if (v > DAC_VDD) v = DAC_VDD;
  return (uint16_t)((v / DAC_VDD) * DAC_MAX + 0.5);
}

void applyDelta(float delta) {
  if (delta >  DELTA_LIMIT) delta =  DELTA_LIMIT;
  if (delta < -DELTA_LIMIT) delta = -DELTA_LIMIT;

  deltaCmd = delta;

  float outA = IDLE_A + deltaCmd + CAL_A;
  float outB = IDLE_B - deltaCmd + CAL_B;

  outA = constrain(outA, 0.0, DAC_VDD);
  outB = constrain(outB, 0.0, DAC_VDD);

  lastCodeA = voltToCode(outA);
  lastCodeB = voltToCode(outB);

  mcp.setChannelValue(MCP4728_CHANNEL_A, lastCodeA);
  mcp.setChannelValue(MCP4728_CHANNEL_B, lastCodeB);

  lastOutA = outA;
  lastOutB = outB;
}

void resetPID() {
  integralTerm = 0.0;
  lastErrorDeg = 0.0;
  lastPTerm    = 0.0;
  lastITerm    = 0.0;
  lastDTerm    = 0.0;
  prevAngleDeg = measuredAngleDeg;

  // Reset slew to current physical angle
  targetSlewed = measuredAngleDeg;
}

void stopOutput() {
  resetPID();
  applyDelta(0.0);
  
}

void enterHold() {
  if (!pidEnabled) {
    pidEnabled = true;
    resetPID();
  }
  holdLatched = true;
  targetAngleDeg = measuredAngleDeg;
}

int steerState() {
  if (!pidEnabled) return 0;
  return holdLatched ? 2 : 1;
}

const char* resetCauseName(uint8_t cause) {
  if (cause & _BV(WDRF))  return "wdt";
  if (cause & _BV(BORF))  return "bod";
  if (cause & _BV(EXTRF)) return "ext";
  if (cause & _BV(PORF))  return "por";
  return "unknown";   // optiboot may clear MCUSR before the sketch runs
}

// =========================
// SETUP
// =========================
void setup() {
  uint8_t resetCause = MCUSR;   // [ESTOP] read the reset cause before anything clears it
  MCUSR = 0;                    // [ESTOP]
    // NEW: first line. After a watchdog reset the watchdog can still be running;
  // disable it so setup is not interrupted. (With the old Nano bootloader a watchdog
  // reset can cause a reboot loop -> use Uno / new Nano bootloader.)
  wdt_disable();

  Serial.begin(115200);
  Serial.setTimeout(20);   // NEW: readStringUntil waits max 20 ms (default 1000 ms would trip the watchdog)
  //delay(300);   //OLD

  // CHANGED: DAC is initialised FIRST so the outputs go to idle as early as possible after a reboot
  Wire.begin();
  Wire.setClock(100000);             // CHANGED: was 400000. 100 kHz is more robust against noise / long wires
  Wire.setWireTimeout(3000, true);   // NEW: I2C gives up after 3 ms instead of hanging forever
                                     //      (needs Arduino AVR Boards 1.8.3 or newer)

  wdt_enable(WDTO_250MS);            // NEW: from here on, a hang > 250 ms reboots the board

  //Wire.begin();
  //Wire.setClock(400000);
  if (!mcp.begin()) {
    Serial.println("ERR,MCP4728_NOT_FOUND");
    while (1) {}
  }

  measuredAngleDeg = getSteeringAngleDeg(0);
  prevAngleDeg     = measuredAngleDeg;
  applyDelta(0.0);
  wdt_reset();

  pinMode(clkPin, INPUT_PULLUP);
  pinMode(dtPin,  INPUT_PULLUP);

  lastState = (digitalRead(clkPin) << 1) | digitalRead(dtPin);

  enableInterrupt(clkPin, handleEncoder, CHANGE);
  enableInterrupt(dtPin,  handleEncoder, CHANGE);

  serialDirectMode = true;

  #if STEERING_USE_CAN   // [ESTOP]
    mcp2515.reset();
    mcp2515.setBitrate(CAN_500KBPS, MCP_8MHZ);
    mcp2515.setNormalMode();
    wdt_reset();
  #endif                 // [ESTOP]

  lastHbReceived = millis();
  lastHbSent     = millis();
  lastSerialRxMs = millis(); 

  Serial.println("READY");
  Serial.print("INFO,RESET,"); Serial.println(resetCauseName(resetCause));   // [ESTOP]
  Serial.println("DATA,t_ms,target_deg,target_slew_deg,angle_deg,error_deg,delta_v,va_out,vb_out,codeA,codeB,count,pid_on,p_term,i_term,d_term,isr,invalid,state");   // [ESTOP] added ",state"
  Serial.println("Serial cmds: e/d=PID on/off | x=hold | h=keepalive | z=zero encoder | r=reverse | s<deg>=set target | kp/ki/kd<val>=tune");   // [ESTOP] added x, h

}


// =========================
// MAIN LOOP
// =========================
void loop() {
  wdt_reset();

  unsigned long now = millis();

#if STEERING_USE_CAN   // [ESTOP]
  if (mcp2515.readMessage(&canMsg) == MCP2515::ERROR_OK) {
    if (!serialDirectMode && canMsg.can_id == CAN_HB_STERFBOARD) {
      heartbeatInterval = (canMsg.data[0] << 8) | canMsg.data[1];
      mode              = canMsg.data[2];
      lastHbReceived    = now;
      Debug.print("Mode: "); Debug.println(mode);

      if ((mode == 1 || mode == 2) && !pidEnabled) {
        resetPID();
        pidEnabled = true;
        Serial.println("INFO,CAN_AUTO_PID_ENABLED");
      }
      else if (mode != 1 && mode != 2 && pidEnabled) {
        pidEnabled = false;
        stopOutput();
        Serial.println("INFO,CAN_PID_DISABLED");
      }
    }
    else if (!serialDirectMode && canMsg.can_id == CAN_STEERING_TARGET) {
      float rxAngle, rxAngleVel;
      memcpy(&rxAngle,    canMsg.data,     4);
      memcpy(&rxAngleVel, canMsg.data + 4, 4);

      if (mode == 1 || mode == 2) {
        if (abs(rxAngle - targetAngleDeg) > 5.0) integralTerm = 0.0;
        targetAngleDeg = rxAngle;
        Debug.print("Target: "); Debug.println(targetAngleDeg, 2);
      }
    }
  }

  if (!serialDirectMode && (now - lastHbReceived > heartbeatInterval * 4)) {
    if (pidEnabled) {
      pidEnabled = false;
      stopOutput();
      Serial.println("WARN,WATCHDOG_NO_HB");
    }
  }
#endif      

  // NEW: serial watchdog -> hold the current angle when the PC stops talking
  if (serialDirectMode && pidEnabled && !holdLatched &&       // [ESTOP] was: !serialLost
      (now - lastSerialRxMs > SERIAL_TIMEOUT_MS)) {
    // [ESTOP] removed: serialLost = true;                    // only act once per loss
    // [ESTOP] removed: targetAngleDeg = measuredAngleDeg;    // hold where the wheel is right now
    enterHold();   // [ESTOP] latched hold at the current angle, integral kept (no output step)
    Serial.println("WARN,SERIAL_LOST_HOLDING");
  }

#if STEERING_USE_CAN   // [ESTOP]
  if (now - lastHbSent >= heartbeatInterval) {
    struct can_frame hb;
    hb.can_id  = CAN_HB_STEERING;
    hb.can_dlc = 8;
    float fDelta = deltaCmd;
    float fAngle = measuredAngleDeg;
    memcpy(hb.data,     &fDelta, 4);
    memcpy(hb.data + 4, &fAngle, 4);
    mcp2515.sendMessage(&hb);
    lastHbSent = now;
  }
#endif  

  if (now - lastControlMs >= controlPeriodMs) {
    lastControlMs = now;
    runControl();
  }

  if (now - lastPrintMs >= printIntervalMs) {
    lastPrintMs = now;
    printDataLine();
  }

  handleSerial();
}

// =========================
// PID CONTROL
// =========================
void runControl() {
  long countsSnapshot;
  noInterrupts();
  countsSnapshot = counter;
  interrupts();

  measuredAngleDeg = getSteeringAngleDeg(countsSnapshot);

  if (!pidEnabled) {
    applyDelta(0.0);
    prevAngleDeg = measuredAngleDeg;
    targetSlewed = measuredAngleDeg;
    integralTerm = 0.0;
    inDeadband = false;
    lastErrorDeg = targetAngleDeg - measuredAngleDeg;
    lastPTerm = lastITerm = lastDTerm = 0.0;
    return;
  }

  float dt = controlPeriodMs / 1000.0f;

  // Slew rate limit the target
  // float maxStep = MAX_TARGET_RATE * dt;
  // if (targetAngleDeg > targetSlewed + maxStep) {
  //   targetSlewed += maxStep;
  // }
  // else if (targetAngleDeg < targetSlewed - maxStep) {
  //   targetSlewed -= maxStep;
  // }
  // else {
  //   targetSlewed = targetAngleDeg;
  // }

  // float rawError = targetSlewed - measuredAngleDeg;
  float rawError = targetAngleDeg - measuredAngleDeg;
  float error = rawError;

  // CHANGED: deadband with hysteresis (replaces the old inErrorDeadband block)
  if (inDeadband) {
    if (fabs(rawError) > DB_EXIT_DEG) inDeadband = false;    // error grew too big -> correct again
  } else {
    if (fabs(rawError) < DB_ENTER_DEG) inDeadband = true;    // close enough -> stop correcting
  }

  if (inDeadband) {
    error = 0.0;   // no P action
    // CHANGED: integral is FROZEN here (old code set it to 0, which caused an output step)
  } else {
    integralTerm += error * dt;
    integralTerm = constrain(integralTerm, -integralLimit, integralLimit);
  }

  // bool inErrorDeadband = fabs(rawError) < ERROR_DEADBAND_DEG;
  // if (inErrorDeadband) {
  //   error = 0.0;
  //   integralTerm = 0.0;
  // } else {
  //   integralTerm += error * dt;
  //   integralTerm = constrain(integralTerm, -integralLimit, integralLimit);
  // }

  float dMeas = (measuredAngleDeg - prevAngleDeg) / dt;

  lastPTerm = Kp * error;
  lastITerm = Ki * integralTerm;
  lastDTerm = -Kd * dMeas;

  float u = lastPTerm + lastITerm + lastDTerm;

  if (fabs(u) < OUTPUT_DEADBAND_V) {
    u = 0.0;
  }

  u = constrain(u, -DELTA_LIMIT, DELTA_LIMIT);

  applyDelta(u);

  lastErrorDeg = error;
  prevAngleDeg = measuredAngleDeg;
}

// =========================
// ENCODER ISR
// =========================
void handleEncoder() {
  uint8_t clkState = digitalRead(clkPin);
  uint8_t dtState  = digitalRead(dtPin);
  uint8_t newState = (clkState << 1) | dtState;

  switch ((lastState << 2) | newState) {
    case 0b0001: case 0b0111: case 0b1110: case 0b1000:
      counter++; isrEvents++; break;
    case 0b0010: case 0b1011: case 0b1101: case 0b0100:
      counter--; isrEvents++; break;
    default:
      if (newState != lastState) invalidTransitions++;
      break;
  }
  lastState = newState;
}

// =========================
// SERIAL PRINT
// =========================
void printDataLine() {
  long c; unsigned long isr, inv;
  noInterrupts(); c = counter; isr = isrEvents; inv = invalidTransitions; interrupts();

  Serial.print("DATA,");  Serial.print(millis());
  Serial.print(",");      Serial.print(targetAngleDeg, 3);
  Serial.print(",");      Serial.print(targetSlewed, 3);
  Serial.print(",");      Serial.print(measuredAngleDeg, 3);
  Serial.print(",");      Serial.print(lastErrorDeg, 3);
  Serial.print(",");      Serial.print(deltaCmd, 4);
  Serial.print(",");      Serial.print(lastOutA, 4);
  Serial.print(",");      Serial.print(lastOutB, 4);
  Serial.print(",");      Serial.print(lastCodeA);
  Serial.print(",");      Serial.print(lastCodeB);
  Serial.print(",");      Serial.print(c);
  Serial.print(",");      Serial.print(pidEnabled ? 1 : 0);
  Serial.print(",");      Serial.print(lastPTerm, 6);
  Serial.print(",");      Serial.print(lastITerm, 6);
  Serial.print(",");      Serial.print(lastDTerm, 6);
  Serial.print(",");      Serial.print(isr);            // <- this line is missing
  Serial.print(",");      Serial.print(inv);            // [ESTOP] was: Serial.println(inv);
  Serial.print(",");      Serial.println(steerState()); // [ESTOP] new column: 0 off, 1 track, 2 hold

}

// =========================
// SERIAL COMMANDS
// =========================
void handleSerial() {
  if (!Serial.available()) return;
  String s = Serial.readStringUntil('\n');
  s.trim();
  if (s.length() == 0) return;

// NEW: any received line counts as "PC is alive"
  lastSerialRxMs = millis();
  // [ESTOP] removed: hold is latched now, receiving lines again must not resume tracking
  // [ESTOP] removed: if (serialLost) {
  // [ESTOP] removed:   serialLost = false;
  // [ESTOP] removed:   Serial.println("INFO,SERIAL_RESTORED");   // control resumes; target stays at the held angle
  // [ESTOP] removed: }
    // NEW: keepalive command, does nothing else
  if (s == "h" || s == "H") return;


  if (s == "e" || s == "E") {
    serialDirectMode = true;
    pidEnabled = true;
    holdLatched = false;
    resetPID();
    targetAngleDeg = measuredAngleDeg;
    Serial.println("INFO,PID_ENABLED");
    return;
  }

  if (s == "d" || s == "D") {
    serialDirectMode = true;
    pidEnabled = false;
    holdLatched = false; 
    stopOutput();
    Serial.println("INFO,PID_DISABLED");
    return;
  }

  if (s == "x" || s == "X") {
    serialDirectMode = true;
    if (steerState() != 2) {
      enterHold();
      Serial.println("INFO,HOLD");
    }
    return;
  }

  if (s == "z" || s == "Z") {
    noInterrupts(); counter = 0; invalidTransitions = 0; isrEvents = 0; interrupts();
    measuredAngleDeg = prevAngleDeg = 0.0;
    targetAngleDeg = 0.0; 
    resetPID();
    Serial.println("INFO,ENCODER_ZEROED");
    return;
  }

  if (s == "p" || s == "P") { printDataLine(); return; }

  if (s == "r" || s == "R") {
    encoderCwRight = !encoderCwRight;
    Serial.print("INFO,ENCODER_REVERSED,"); Serial.println(encoderCwRight ? 1 : 0);
    return;
  }

  if (s.startsWith("s")) {
    serialDirectMode = true;
    if (!pidEnabled || holdLatched) return;
    float t = s.substring(1).toFloat();
    t = constrain(t, -620.0f, 620.0f);

    if (fabs(t - targetAngleDeg) > 5.0f) {
      integralTerm = 0.0f;
    }

    targetAngleDeg = t;
    Serial.print("INFO,TARGET_DEG,"); Serial.println(targetAngleDeg, 3);
    return;
  }

  if (s.startsWith("kp")) { Kp = s.substring(2).toFloat(); Serial.print("INFO,KP,"); Serial.println(Kp, 6); return; }
  if (s.startsWith("ki")) { Ki = s.substring(2).toFloat(); Serial.print("INFO,KI,"); Serial.println(Ki, 6); return; }
  if (s.startsWith("kd")) { Kd = s.substring(2).toFloat(); Serial.print("INFO,KD,"); Serial.println(Kd, 6); return; }
}
