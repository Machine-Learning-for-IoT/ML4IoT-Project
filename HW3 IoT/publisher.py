""" HW3 - Exercise 1.1 """
import paho.mqtt.client as mqtt
from datetime import datetime
from time import sleep
import uuid
import time
import psutil
import json

# Create a new MQTT client
client = mqtt.Client()

# Connect to the MQTT broker
client.connect('mqtt.eclipseprojects.io', 1883)

mac_address = hex(uuid.getnode())
events_list = list()  #it is a list of dictionaries and contains the events
topic = 's316001'

# Publish a message to a topic. Use your studend ID as topic
while True:
    
    timestamp = time.time()
    timestamp_ms = int(timestamp * 1000)
    battery_level = psutil.sensors_battery().percent
    power_plugged = int(psutil.sensors_battery().power_plugged)

    event = {'timestamp': timestamp_ms, 'battery_level': battery_level, 'power_plugged': power_plugged}
    events_list.append(event)
    
    if len(events_list) == 10: #I collected 10 consecutive records
        message = {'mac_address':mac_address, 'events' : events_list}   #create the message to be sent
        json_message = json.dumps(message)   #convert the message in the json format
        client.publish(topic, json_message)  #publish the message
        #print(message)
        
        events_list.clear()  #remove all the elements from the list

    time.sleep(1)  #since the monitoring should be done with a sampling rate of 1 second
    

