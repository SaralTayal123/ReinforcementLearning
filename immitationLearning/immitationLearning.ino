#include <Servo.h>

//const int mode = 13;
const int rx = 3;
Servo steering;
Servo motor;
const int accel = 12;
int lastpwm = 0;
void setup() {
  pinMode(10, INPUT); //rpi control
  pinMode(13, OUTPUT);
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
  int pwm = 1;
  digitalWrite(13 ,LOW);
  if (digitalRead(10) == HIGH){    
    if (Serial.available()) {
//      String data = Serial.readStringUntil('\n');
//      pwm = data.toInt();
      pwm = Serial.read() - '0';
      digitalWrite(13 ,HIGH);
      steering.write(pwm); //write RPI values
      lastpwm = pwm;
    }
  }
  
  //RC system in control
  else{
    int pwmController = pulseIn(rx, HIGH); //read PWM value
    Serial.println(map(pwmController, 1350, 2000, 0, 180)); //Send to RPI
    steering.write(map(pwmController, 1350, 2000, 0, 180)); //write RPI values
    delay(10);
  }
  
}
