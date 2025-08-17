#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define DEFAULT_FREQ 50   // Hz
#define MIN_US 400        
#define MAX_US 2400       
#define MID_US 1400
#define MIN_ANGLE -90.0   
#define MAX_ANGLE 90.0    
#define DEG_TO_RAD 0.01745329251
#define PI 3.14159265358979323846
bool debugMode;
bool started;

struct JointPos {
  float x;
  float y;
  float z;
  float angle; // in degrees
};

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
int freqHz = DEFAULT_FREQ;

uint16_t lastTicks[16] = {0};
uint16_t lastUs[16]    = {0};
float lastAngles[16]   = {0};


uint16_t angleToUs(float angle) {
  // Map -90..+90 to MIN_US..MAX_US
  return MIN_US + (uint16_t)(((angle + 90.0) / 180.0) * (MAX_US - MIN_US));
}

float usToAngle(uint16_t us) {
  // Map MIN_US..MAX_US to -90..+90
  float t = float(us - MIN_US) / float(MAX_US - MIN_US);
  return (t * 180.0) - 90.0;
}

uint16_t usToTicks(uint16_t us, int freq) {
  // ticks = us * freq * 4096 / 1e6
  uint32_t t = (uint32_t)us * (uint32_t)freq * 4096UL;
  t /= 1000000UL;
  if (t > 4095) t = 4095;
  return (uint16_t)t;
}

// Calculate joint positions for the arm (base, shoulder, elbow, wrist, hand)
void calcJointPositions(const float angles[16], JointPos outJoints[5]) {
  // Link lengths (meters)
  const float L1 = 0.090; // shoulder to elbow
  const float L2 = 0.090; // elbow to wrist
  const float L3 = 0.020; // wrist to hand

  // Convert angles to radians
  float baseRad     = angles[15] * DEG_TO_RAD;
  float shoulderRad = angles[14] * DEG_TO_RAD;
  float elbowRad    = angles[13] * DEG_TO_RAD;
  float wristRad    = angles[12] * DEG_TO_RAD;
  // float handRad     = angles[11] * DEG_TO_RAD; // optional

  // Base position (origin)
  outJoints[0].x = 0;
  outJoints[0].y = 0;
  outJoints[0].z = 0;
  outJoints[0].angle = angles[15];

  // Shoulder position (same as base, but angle for reference)
  outJoints[1].x = 0;
  outJoints[1].y = 0;
  outJoints[1].z = 0;
  outJoints[1].angle = angles[14];

  // Elbow position
  float x1 = L1 * cos(shoulderRad) * cos(baseRad);
  float y1 = L1 * cos(shoulderRad) * sin(baseRad);
  float z1 = L1 * sin(shoulderRad);
  outJoints[2].x = x1;
  outJoints[2].y = y1;
  outJoints[2].z = z1;
  outJoints[2].angle = angles[13];

  // Wrist position
  float totalShoulder = shoulderRad + elbowRad;
  float x2 = x1 + L2 * cos(totalShoulder) * cos(baseRad);
  float y2 = y1 + L2 * cos(totalShoulder) * sin(baseRad);
  float z2 = z1 + L2 * sin(totalShoulder);
  outJoints[3].x = x2;
  outJoints[3].y = y2;
  outJoints[3].z = z2;
  outJoints[3].angle = angles[12];

  // Hand position
  float totalWrist = totalShoulder + wristRad;
  float x3 = x2 + L3 * cos(totalWrist) * cos(baseRad);
  float y3 = y2 + L3 * cos(totalWrist) * sin(baseRad);
  float z3 = z2 + L3 * sin(totalWrist);
  outJoints[4].x = x3;
  outJoints[4].y = y3;
  outJoints[4].z = z3;
  outJoints[4].angle = angles[11];
}

// Validation function: checks if new angle is valid given current arm state
bool validation(int ch, float newAngle) {

  //define link lengths (meters)
  const float L1 = 0.090; // shoulder to elbow
  const float L2 = 0.090; // elbow to wrist
  const float L3 = 0.020; // wrist to hand

  // Copy current angles
  float angles[16];
  for (int i = 0; i < 16; ++i) angles[i] = lastAngles[i];

  // Update the candidate joint with the new angle
  angles[ch] = newAngle;

  // Calculate joint positions
  //JointPos joints[5];
  //calcJointPositions(angles, joints);

  float baseAngle = angles[15];
  float shoulderAngle = angles[14];
  float elbowAngle = angles[13];
  float wristAngle = angles[12];

  // --- New constraint --- dont drive into the structure
  if (baseAngle >= -45.0 && baseAngle <= 45.0) { // overhangs structure
    if (shoulderAngle <= -75) {
      if (elbowAngle < 0) {
        Serial.println("ERR: into back structure");
        return false; // Invalid elbow angle
      }
    }
  }
 
  // --- New constraint --- dont bend wrist into bicep
  // wrist cannot be negative if wlbow is less than 
  if (elbowAngle > 45) {
    if (wristAngle < 0) {
      Serial.println("ERR: into bicep");
      return false; // Invalid wrist angle
    }
  }
 
  // --- New constraint --- dont shoulder drive hand into base
  if (shoulderAngle > 0) {
    if (elbowAngle > 50) {
      Serial.println("ERR: into base");
      return false; // Invalid wrist angle
    }
  }

  // --- New constraint --- dont drive hand into floor
  float shoulderAngleRad = shoulderAngle * DEG_TO_RAD;
  float elbowAngleRad = elbowAngle * DEG_TO_RAD;

  // z1: vertical position from elbow joint to z=0
  float z1 = L1 * sin(shoulderAngleRad);

  // z2: vertical position from elbow joint to wrist joint
  float z2 = L2 * sin(PI / 2 - (shoulderAngleRad + elbowAngleRad));

  // zWrist: vertical position of the wrist from z=0
  float zWrist = z1 - z2;

  // Constraint: hand must not be farther than -0.075 m from the shoudler joint
  if (zWrist > 0.080) {
    Serial.println("ERR: Hand too close to floor");
    return false;
  }
  

  // ...other constraints...

  lastAngles[ch] = newAngle;
  applyPulse(ch, newAngle);
  return true;
}

void applyPulse(int ch, float angle) {
  uint16_t us = angleToUs(angle);
  us = constrain(us, MIN_US, MAX_US);
  uint16_t ticks = usToTicks(us, freqHz);
  pwm.setPWM(ch, 0, ticks);
  lastUs[ch] = us;
  lastTicks[ch] = ticks;
}

void sendLine(const String &s) {
  Serial.println(s);
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  // Uppercase first token
  int sp1 = line.indexOf(' ');
  String cmd = (sp1 >= 0) ? line.substring(0, sp1) : line;
  cmd.toUpperCase();

  if (cmd == "PING") {
    sendLine("PONG");
    return;
  }

  if (cmd == "F") {
    // F <freqHz>
    if (sp1 < 0) { sendLine("ERR FREQ"); return; }
    int newF = line.substring(sp1 + 1).toInt();
    if (newF < 24 || newF > 1000) { sendLine("ERR FREQ_RANGE"); return; }
    freqHz = newF;
    pwm.setPWMFreq(freqHz);
    sendLine(String("OK F ") + freqHz);
    return;
  }

  if (cmd == "GET") {
    // GET <ch>
    if (sp1 < 0) { sendLine("ERR ARGS"); return; }
    int ch = line.substring(sp1 + 1).toInt();
    if (ch < 0 || ch > 15) { sendLine("ERR CH"); return; }
    sendLine(String("OK GET ") + ch + " " + lastUs[ch] + " " + lastTicks[ch]);
    return;
  }

  if (cmd == "BUG") {
    // debug <on|off>
    if (sp1 < 0) { sendLine("ERR ARGS"); return; }
    String mode = line.substring(sp1 + 1);
    mode.toLowerCase();
    if (mode == "on") {
      debugMode = true;
      sendLine("DEBUG ON");
    } else if (mode == "off") {
      debugMode = false;
      sendLine("DEBUG OFF");
    } else {
      sendLine("ERR INVALID DEBUG MODE");
    }
    return;
  }

  if (cmd == "P") {
    // P <ch> <us>
    if (sp1 < 0) { sendLine("ERR ARGS"); return; }
    int sp2 = line.indexOf(' ', sp1 + 1);
    if (sp2 < 0) { sendLine("ERR ARGS"); return; }

    int ch = line.substring(sp1 + 1, sp2).toInt();
    long us = line.substring(sp2 + 1).toInt();

    if (ch < 0 || ch > 15) { sendLine("ERR CH"); return; }
    if (us < MIN_US || us > MAX_US) { sendLine("ERR US"); return; }
    float angle = usToAngle((uint16_t)us);
    
    if (!debugMode) {
      if (!validation(ch, angle)) {
        sendLine("ERR INVALID P");
        return;
      }
    } 
    
    else {
      // In debug mode, just apply the pulse without validation
      applyPulse(ch, angle);
      lastUs[ch] = us;
      lastAngles[ch] = angle;
      lastTicks[ch] = usToTicks((uint16_t)us, freqHz);
    }
    
    sendLine(String("OK P ") + ch + " " + angle);
    return;
  }

  if (cmd == "A") {
    // A <ch> <angle>
    int sp2 = line.indexOf(' ', sp1 + 1);
    if (sp2 < 0) { sendLine("ERR ARGS"); return; }

    int ch = line.substring(sp1 + 1, sp2).toInt();
    float angle = line.substring(sp2 + 1).toFloat();

    if (ch < 0 || ch > 15) { sendLine("ERR CH"); return; }
    if (angle < MIN_ANGLE || angle > MAX_ANGLE) { sendLine("ERR ANGLE"); return; }

    if (!debugMode) {
      if (!validation(ch, angle)) {
        sendLine("ERR INVALID A");
        return;
      }
    } 
    
    else {
      // In debug mode, just apply the angle without validation
      applyPulse(ch, angle);
      lastUs[ch] = angleToUs(angle);
      lastAngles[ch] = angle;
      lastTicks[ch] = usToTicks(lastUs[ch], freqHz);
    }

    sendLine(String("OK A ") + ch + " " + angle);
    return;
  }
  

  sendLine("ERR CMD " + cmd);
}



void setup() {  

  pwm.begin();
  pwm.setPWMFreq(freqHz);

  Serial.begin(9600);
  while (!Serial) { ; }

  

  //move to home position
  handleCommand("A 14 -90"); // shoulder
  delay(2000); // wait for servo to move
  handleCommand("A 13 90"); // elbow
  handleCommand("A 15 0"); // base
  handleCommand("A 12 0"); // wrist
  handleCommand("A 11 0"); // hand


  sendLine("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }
}