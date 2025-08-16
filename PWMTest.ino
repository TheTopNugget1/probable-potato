#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define DEFAULT_FREQ 50   // Hz
#define MIN_US 400        // safety clamp
#define MAX_US 2800       // safety clamp

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
int freqHz = DEFAULT_FREQ;

uint16_t lastTicks[16] = {0};
uint16_t lastUs[16]    = {0};

uint16_t usToTicks(uint16_t us, int freq) {
  // ticks = us * freq * 4096 / 1e6
  uint32_t t = (uint32_t)us * (uint32_t)freq * 4096UL;
  t /= 1000000UL;
  if (t > 4095) t = 4095;
  return (uint16_t)t;
}

void applyPulse(int ch, uint16_t us) {
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

  if (cmd == "P") {
    // P <ch> <us>
    if (sp1 < 0) { sendLine("ERR ARGS"); return; }
    int sp2 = line.indexOf(' ', sp1 + 1);
    if (sp2 < 0) { sendLine("ERR ARGS"); return; }

    int ch = line.substring(sp1 + 1, sp2).toInt();
    long us = line.substring(sp2 + 1).toInt();

    if (ch < 0 || ch > 15) { sendLine("ERR CH"); return; }
    if (us <= 0) { sendLine("ERR US"); return; }

    applyPulse(ch, (uint16_t)us);
    sendLine(String("OK P ") + ch + " " + lastUs[ch] + " " + lastTicks[ch]);
    return;
  }
  
  sendLine("ERR CMD");
}

void setup() {
  Serial.begin(9600);
  while (!Serial) { ; }
  pwm.begin();
  pwm.setPWMFreq(freqHz);
  // Optional: initialize all channels to a safe center (1500 us)
  for (int ch = 0; ch < 16; ++ch) {
    applyPulse(ch, 1500);
  }
  sendLine("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }
}