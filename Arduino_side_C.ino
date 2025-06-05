#include <Servo.h>

const int numServos = 3;  // Number of servos in use
Servo servos[numServos];   // Array of servo objects
const int servoPins[numServos] = {2, 3, 4};    // Define servo pins

void setup() {
    Serial.begin(115200);
    for (int i = 0; i < numServos; i++) {
        servos[i].attach(servoPins[i]);
        servos[i].write(0);
    }
}

void loop() {
    if (Serial.available() > 0) {  
      String message = Serial.readStringUntil('\n');  // Read full command
      processCommand(message);  // Process command
    }
}

void processCommand(String command) {
    command.trim();  // Remove leading/trailing whitespace
    if (command.length() == 0) return;

    // New: Parse "1:180 2:180 3:180"
    int lastSpace = 0;
    int servoSetCount = 0;
    while (servoSetCount < numServos) {
        int spaceIndex = command.indexOf(' ', lastSpace);
        String part;
        if (spaceIndex == -1) {
            part = command.substring(lastSpace);
        } else {
            part = command.substring(lastSpace, spaceIndex);
        }
        int colonIndex = part.indexOf(':');
        if (colonIndex == -1) break;
        int servoID = part.substring(0, colonIndex).toInt();
        int angle = part.substring(colonIndex + 1).toInt();
        if (servoID >= 1 && servoID <= numServos && angle >= 0 && angle <= 360) {
            servos[servoID - 1].write(constrain(angle, 0, 180)); // Clamp to servo range
        }
        servoSetCount++;
        if (spaceIndex == -1) break;
        lastSpace = spaceIndex + 1;
    }
}
