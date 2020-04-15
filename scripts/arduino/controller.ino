#include <Time.h>
#include <Wire.h>
#include <DS1307RTC.h>  // a basic DS1307 library that returns time as a time_t
#include <TimeAlarms.h>

int redPin1 = 3;
int greenPin1 = 5;
int bluePin1 = 6;
int redPin2 = 9;
int greenPin2 = 10;
int bluePin2 = 11;
int motorPin1 = 4;
int motorPin2 = 8;
int feeder;

int lastFour[] = {0, 1, 0, 1};
boolean same;

void setup()  {
  randomSeed(analogRead(0));
  pinMode(motorPin1, OUTPUT);             //set the motor pin as output
  pinMode(motorPin2, OUTPUT);             //set the motor pin as output
  for (int i = 0; i < 3; i++) {
    pinMode(redPin1, OUTPUT);
    pinMode(greenPin1, OUTPUT);
    pinMode(bluePin1, OUTPUT);
    pinMode(redPin2, OUTPUT);
    pinMode(greenPin2, OUTPUT);
    pinMode(bluePin2, OUTPUT);   //Set the three LED pins as outputs
  }
  setColor1(0, 0, 0); //Turn off led 1
  setColor2(0, 0, 0); //Turn off led 2
  Serial.begin(9600);
  while (!Serial) ; // wait until Arduino Serial Monitor opens
  setSyncProvider(RTC.get);   // the function to get the time from the RTC
  if (timeStatus() != timeSet)
    Serial.println("Unable to sync with the RTC");
  else
    Serial.println("RTC has set the system time");
  Alarm.alarmRepeat(8, 30, 0, Alarm1);
  Alarm.alarmRepeat(11, 30, 0, Alarm2);
  Alarm.alarmRepeat(14, 30, 00, Alarm3);
  Alarm.alarmRepeat(17, 30, 0, Alarm4);

}

void loop()
{
  if (timeStatus() == timeSet) {
    digitalClockDisplay();


  } else {
    Serial.println("The time has not been set.  Please run the Time");
    Serial.println("TimeRTCSet example, or DS1307RTC SetTime example.");
    Serial.println();
    delay(4000);
  }
  Alarm.delay(1000);
}

void digitalClockDisplay() {
  // digital clock display of the time
  Serial.print(hour());
  printDigits(minute());
  printDigits(second());
  Serial.print(" ");
  Serial.print(day());
  Serial.print(" ");
  Serial.print(month());
  Serial.print(" ");
  Serial.print(year());
  Serial.println();
}

void printDigits(int digits) {
  // utility function for digital clock display: prints preceding colon and leading 0
  Serial.print(":");
  if (digits < 10)
    Serial.print('0');
  Serial.print(digits);
}

void Alarm1() {
  runAll();
}
void Alarm2() {
  runAll();
}
void Alarm3() {
  runAll();
}
void Alarm4() {
  runAll();
}

void setColor1(int red1, int green1, int blue1)
{
  analogWrite(redPin1, red1);
  analogWrite(greenPin1, green1);
  analogWrite(bluePin1, blue1);
}
void setColor2(int red2, int green2, int blue2)
{  analogWrite(redPin2, red2);
  analogWrite(greenPin2, green2);
  analogWrite(bluePin2, blue2);
}

void runAll() {
  same = 1;
  while (same == 1) { // we go through the while loop at least once to ensure that at least one value in lastFour[] array != newValue,
    // which is randomly generated within the while loop
    // we keep looping until 'newValue' is not the same as the values in the lastFour[] array
    feeder = random(2); // initialize 'newValue' with a random value of 0,1
    for (int index = 0; index < 4; index++) { // step through lastFour[] array and compare each value with 'newValue'
      // if a value in array isn't equal to newValue, set 'same' to False
      if (feeder != lastFour[index]) {
        same = 0;
      }
    } // end for (int index = 0; index < 4; index++)
  } // end while (same == 1)

  // we have ensured that at least one value in lastFour[] array doesn't equal 'newValue'
  // now update the lastFour[] array and use 'newValue'
  lastFour[0] = lastFour[1];
  lastFour[1] = lastFour[2];
  lastFour[2] = lastFour[3];
  lastFour[3] = feeder;
  for (int i = 0; i < 4; i++) {
    Serial.print(lastFour[i]);
    Serial.print(", ");
  }
  Serial.println("");
  if (feeder == 0)
  {
    setColor1(255, 60, 0); //This one is the color that will feed
    setColor2(0, 255, 255); // Does not feed
    delay(3000);
    setColor1(0,0,0);
    setColor2(0,0,0);
    delay(1000);
    digitalWrite(motorPin1, HIGH);
    delay(100);
    digitalWrite(motorPin1, LOW);
    delay(100);
    digitalWrite(motorPin1, HIGH);
    delay(100);
    digitalWrite(motorPin1, LOW);
  }
  else if (feeder == 1)
  {
    setColor(0,255,255); //Negative
    setColor(255,60,0); // Positive. This one is the color that will feed
    delay(3000);
    setColor(0,0,0);
    setColor(0,0,0);
    delay(1000);
    digitalWrite(motorPin2, HIGH);
    delay(100);
    digitalWrite(motorPin2, LOW);
    delay(100);
    digitalWrite(motorPin2, HIGH);
    delay(100);
    digitalWrite(motorPin2, LOW);
  }
}
}
