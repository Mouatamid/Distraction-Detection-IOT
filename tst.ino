#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include "SDHT.h"

SDHT dht;


// Change the credentials below, so your ESP8266 connects to your router
const char* ssid = "HUAWEIY9";
const char* password = "laalejahmed";

// Change the variable to your Raspberry Pi IP address, so it connects to your MQTT broker
const char* mqtt_server = "192.168.43.205";

// Initializes the espClient
WiFiClient espClient;
PubSubClient client(espClient);

// Connect an LED to each GPIO of your ESP8266
const int ledGPIO5 = D4;
const int ledGPIO4 = 0;
const int ledGPIO6 = D5;
const int ledGPIO7 = D6;
const int ledGPIO8 = D7;
const int ledGPIO9 = D8;
const int ledGPIO10 = D0;
const int ledGPIO11 = D1;
const int ledGPIO12 = D2;
// DHT Sensor


// Initialize DHT sensor.

// Timers auxiliar variables
long now = millis();
long lastMeasure = 0;

// Don't change the function below. This functions connects your ESP8266 to your router
void setup_wifi() {
  delay(10);
  // We start by connecting to a WiFi network
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("WiFi connected - ESP IP address: ");
  Serial.println(WiFi.localIP());
}

// This functions is executed when some device publishes a message to a topic that your ESP8266 is subscribed to
// Change the function below to add logic to your program, so when a device publishes a message to a topic that 
// your ESP8266 is subscribed you can actually do something
void callback(String topic, byte* message, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  String messageTemp;
  
  for (int i = 0; i < length; i++) {
    Serial.print((char)message[i]);
    messageTemp += (char)message[i];
  }
  Serial.println();

  // Feel free to add more if statements to control more GPIOs with MQTT

  // If a message is received on the topic home/office/esp1/gpio2, you check if the message is either 1 or 0. Turns the ESP GPIO according to the message
  if(topic=="esp8266/c"){
      Serial.print("Changing GPIO 3 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO4, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO4, LOW);
        Serial.print("Off");
      }
  }
  if(topic=="esp8266/c1"){
      Serial.print("Changing GPIO 4 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO5, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO5, LOW);
        Serial.print("Off");
      }
  }
  if(topic=="esp8266/c2"){
      Serial.print("Changing GPIO 5 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO6, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO6, LOW);
        Serial.print("Off");
      }
  }
  if(topic=="esp8266/c3"){
      Serial.print("Changing GPIO 6 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO7, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO7, LOW);
        Serial.print("Off");
      }
  }
  if(topic=="esp8266/c4"){
      Serial.print("Changing GPIO 7 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO8, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO8, LOW);
        Serial.print("Off");
      }
  }
  if(topic=="esp8266/c5"){
      Serial.print("Changing GPIO 9 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO9, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO9, LOW);
        Serial.print("Off");
      }
  }
  if(topic=="esp8266/c6"){
      Serial.print("Changing GPIO 10 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO10, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO10, LOW);
        Serial.print("Off");
      }
  }
  if(topic=="esp8266/c7"){
      Serial.print("Changing GPIO 10 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO11, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO11, LOW);
        Serial.print("Off");
      }
  }
  if(topic=="esp8266/c8"){
      Serial.print("Changing GPIO 10 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO12, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO12, LOW);
        Serial.print("Off");
      }
  }
  /*if(topic=="esp8266/c9"){
      Serial.print("Changing GPIO 10 to ");
      if(messageTemp == "1"){
        digitalWrite(ledGPIO12, HIGH);
        Serial.print("On");
      }
      else if(messageTemp == "0"){
        digitalWrite(ledGPIO12, LOW);
        Serial.print("Off");
      }
  }*/
  
  Serial.println();
}

// This functions reconnects your ESP8266 to your MQTT broker
// Change the function below if you want to subscribe to more topics with your ESP8266 
void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
     /*
     YOU  NEED TO CHANGE THIS NEXT LINE, IF YOU'RE HAVING PROBLEMS WITH MQTT MULTIPLE CONNECTIONS
     To change the ESP device ID, you will have to give a unique name to the ESP8266.
     Here's how it looks like now:
       if (client.connect("ESP8266Client")) {
     If you want more devices connected to the MQTT broker, you can do it like this:
       if (client.connect("ESPOffice")) {
     Then, for the other ESP:
       if (client.connect("ESPGarage")) {
      That should solve your MQTT multiple connections problem

     THE SECTION IN loop() function should match your device name
    */
    if (client.connect("ESP8266Client1")) {
      Serial.println("connected");  
      // Subscribe or resubscribe to a topic
      // You can subscribe to more topics (to control more LEDs in this example)
      client.subscribe("esp8266/c");
      client.subscribe("esp8266/c1");
      client.subscribe("esp8266/c2");
      client.subscribe("esp8266/c3");
      client.subscribe("esp8266/c4");
      client.subscribe("esp8266/c5");
      client.subscribe("esp8266/c6");
      client.subscribe("esp8266/c7");
      client.subscribe("esp8266/c8");
      client.subscribe("esp8266/c9");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

// The setup function sets your ESP GPIOs to Outputs, starts the serial communication at a baud rate of 115200
// Sets your mqtt broker and sets the callback function
// The callback function is what receives messages and actually controls the LEDs
void setup() {
  pinMode(ledGPIO4, OUTPUT);
  pinMode(ledGPIO5, OUTPUT);
  pinMode(ledGPIO6, OUTPUT);
  pinMode(ledGPIO7, OUTPUT);
  pinMode(ledGPIO8, OUTPUT);
  pinMode(ledGPIO9, OUTPUT);
  pinMode(ledGPIO10, OUTPUT);
  pinMode(ledGPIO11, OUTPUT);
  pinMode(ledGPIO12, OUTPUT);

  
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

// For this project, you don't need to change anything in the loop function. 
// Basically it ensures that you ESP is connected to your broker
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  if(!client.loop())
     /*
     YOU  NEED TO CHANGE THIS NEXT LINE, IF YOU'RE HAVING PROBLEMS WITH MQTT MULTIPLE CONNECTIONS
     To change the ESP device ID, you will have to give a unique name to the ESP8266.
     Here's how it looks like now:
       client.connect("ESP8266Client");
     If you want more devices connected to the MQTT broker, you can do it like this:
       client.connect("ESPOffice");
     Then, for the other ESP:
       client.connect("ESPGarage");
      That should solve your MQTT multiple connections problem

     THE SECTION IN recionnect() function should match your device name
    */
    client.connect("ESP8266Client1");
    
  //.now = millis();
  // Publishes new temperature and humidity every 10 seconds
  /*if (dht.read(DHT11, D6)) {
    delay(1000);
    lastMeasure = now;
    // Sensor readings may also be up to 2 seconds 'old' (its a very slow sensor)
    String h= String(double(dht.humidity) / 10, 1);
    // Read temperature as Celsius (the default)
    String t = String(double(dht.celsius) / 10, 1);
    String m = "{\"temperature\" : "+t+",\"humidity\" :"+h+"}";
    

char msg[50];
m.toCharArray(msg,50);
    
    client.publish("/esp8266/1/dhtreadings", msg);
    
    Serial.print("Humidity: ");
    Serial.print(h);
    Serial.print(" %\t Temperature: ");
    Serial.print(t);
    Serial.println(" *C ");
    
  }*/
}
