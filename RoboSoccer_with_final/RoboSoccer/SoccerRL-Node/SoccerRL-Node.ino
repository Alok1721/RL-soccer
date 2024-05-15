#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>

// Motor A
int motor1Pin1 = 14;    // GPIO5 (D1 on NodeMCUA) - Pin 1 of L293D for Motor A
int motor1Pin2 = 12;    // GPIO4 (D2 on NodeMCU) - Pin 2 of L293D for Motor A
int enableMotor1 = 13;  // GPIO0 (D3 on NodeMCU) - Pin 3 of L293D for Motor

// Motor B
int motor2Pin1 = 0;    // GPIO2 (D4 on NodeMCU) - Pin 4 of L293D for Motor B
int motor2Pin2 = 4;    // GPIO14 (D5 on NodeMCU) - Pin 5 of L293D for Motor B
int enableMotor2 = 2;  // GPIO12 (D6 on NodeMCU) - Pin 6 of L293D for Motor B


const uint16_t port = 8080;
const char* host = "172.17.7.69";

float baseSpeed = 100;
float stepSpeed = 20;
float forwardDelay = 100;
float backwardDelay = 100;
float leftRotateDelay = 100;
float rightRotateDelay = 80;
float rotateFactor = 0.2;
float resetDelay = 50;
float ConfigArray[8];
String id = "12";

WiFiClient client;
void setup() {
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enableMotor1, OUTPUT);

  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
  pinMode(enableMotor2, OUTPUT);

  Serial.begin(115200);
  Serial.println("Connecting...\n");
  WiFi.mode(WIFI_STA);
  WiFi.begin("RamanLab", "RaMaN@2020");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  while (!client.connect(host, port)) {
    Serial.println("Connection to host failed");
    delay(200);
  }
  Serial.println("Connected to server successful!");

  client.println("ID:" + id);


  delay(250);
}

void GetArray(String encodedList, float* integerArray) {
  int n = encodedList.length();
  int index = 0;
  String tempNum = "";
  for (int i = 0; i < n; i++) {
    char c = encodedList.charAt(i);
    if (c == ' ') {

      integerArray[index++] = tempNum.toFloat();

      tempNum = "";
    } else {

      tempNum += c;
    }
  }
}

void MoveForward() {
  Serial.println("Forward");
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  analogWrite(enableMotor1, baseSpeed);


  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);

  analogWrite(enableMotor2, baseSpeed);
  delay(forwardDelay);
  ResetMotors();
}

void MoveBackward() {
  Serial.println("Backward");
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  analogWrite(enableMotor1, baseSpeed);


  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  analogWrite(enableMotor2, baseSpeed);
  delay(backwardDelay);
  ResetMotors();
}
void RotateRight() {
  Serial.println("Right");
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  analogWrite(enableMotor1, baseSpeed * rotateFactor);


  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  analogWrite(enableMotor2, baseSpeed);
  delay(rightRotateDelay);
  ResetMotors();
}

void RotateLeft() {
  Serial.println("Left");
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  analogWrite(enableMotor1, baseSpeed);


  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  analogWrite(enableMotor2, baseSpeed * rotateFactor);
  delay(leftRotateDelay);
  ResetMotors();
}
void ResetMotors() {
  Serial.println("Reset");
  analogWrite(enableMotor1, 0);
  analogWrite(enableMotor2, 0);
  delay(resetDelay);
}



void loop() {

  while (client.available() > 0) {


    String encodedList = client.readStringUntil('\n');

    if (encodedList[0] == 'C') {
      GetArray(encodedList.substring(1), ConfigArray);
      baseSpeed = ConfigArray[0];
      stepSpeed = ConfigArray[1];
      forwardDelay = ConfigArray[2];
      backwardDelay = ConfigArray[3];
      leftRotateDelay = ConfigArray[4];
      rightRotateDelay = ConfigArray[5];
      rotateFactor = ConfigArray[6];
      resetDelay = ConfigArray[7];
      Serial.println(rotateFactor);
      encodedList="";

    } 
    else if (encodedList[0] == 'A') {
      float Actions[3];
      GetArray(encodedList.substring(1), Actions);
      Serial.println(encodedList.substring(1));
      Serial.println(Actions[2]);
      Serial.println("Got Action");
      int i=0;
      for(i=0;i<3;i++)
      Serial.println(Actions[i]);

      if (Actions[2]) {
        if (Actions[2] == 1)
          RotateLeft();
        else
          RotateRight();
      }
     
      if (Actions[0]) {
        if (Actions[0] == 1)
          MoveForward();
        else
          MoveBackward();
      }
       else
      {
        ResetMotors();
      }
      encodedList="";
    }
    else
    {
      ResetMotors();
    }
  }
}