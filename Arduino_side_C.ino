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

    if (command.startsWith("POS ")) {
        // Format: POS x.xxx,y.yyy,z.zzz
        command = command.substring(4);  // Remove "POS "
        
        float values[3];
        int lastIndex = 0;
        for (int i = 0; i < 3; i++) {
            int commaIndex = command.indexOf(',', lastIndex);
            if (commaIndex == -1 && i < 2) return;  // Not enough values

            String part = (i < 2) ? command.substring(lastIndex, commaIndex)
                                  : command.substring(lastIndex);

            values[i] = part.toFloat();
            lastIndex = commaIndex + 1;
        }

        // Use values[0] (x), values[1] (y), values[2] (z)
        Serial.print("Parsed POS: ");
        Serial.print(values[0], 3); Serial.print(", ");
        Serial.print(values[1], 3); Serial.print(", ");
        Serial.println(values[2], 3);

        // Example: Map these to servo angles
        for (int i = 0; i < numServos; i++) {
            int angle = map(values[i] * 10000, -500, 500, 0, 180);  // Adjust scaling as needed
            angle = constrain(angle, 0, 180);
            servos[i].write(angle);
        }

        return;
    }

    // Handle existing servo command: "servoID:instruction"
    int separatorIndex = command.indexOf(':');
    if (separatorIndex == -1) return;

    int servoID = command.substring(0, separatorIndex).toInt();
    String instruction = command.substring(separatorIndex + 1);

    int inst = instruction.toInt();  // Convert to integer
    if (servoID >= 1 && servoID <= numServos && inst >= 0 && inst <= 180) {
        int servoIndex = servoID - 1;
        servos[servoIndex].write(inst);
    }
}
