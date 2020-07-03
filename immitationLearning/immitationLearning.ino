#include <Servo.h>

const int mode = 13;
const int rx = 3;
Servo steering;
Servo motor;
const int accel = 12;
int lastpwm = 0;
void setup() {
  pinMode(13, INPUT);
  pinMode(5, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(3, INPUT);
  steering.attach(5);
  motor.attach(12);
  Serial.begin(9600);
}
void loop() {
  motor.write(20);  
  //RPI in control
  if (digitalRead(13) == LOW){
    if (Serial.available() > 0) {
      String data = Serial.readStringUntil('\n');
      int pwm = data.toInt(); //read RPI values
//      Serial.println(pwm);
      steering.write(pwm); //write RPI values
      lastpwm = pwm;
    }
  }
  
  //RC system in control
  else{
    int pwmController = pulseIn(rx, HIGH); //read PWM value
    Serial.println(pwmController); //Send to RPI
    steering.write(pwmController); //write RPI values
  }
  
}
