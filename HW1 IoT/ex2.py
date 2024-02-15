import psutil
import redis
import time
import uuid
from datetime import datetime

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--host', type = str, default= '')
parser.add_argument('--port', type = int, default= 0)
parser.add_argument('--user', type = str, default='')
parser.add_argument('--password', type = str, default='')


args = parser.parse_args()

host = args.host
port = args.port
user = args.user
password = args.password

REDIS_HOST = host
REDIS_PORT = port
REDIS_USERNAME = user
REDIS_PASSWORD = password



redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)

mac_address = hex(uuid.getnode())

try:
    redis_client.ts().create(f'{mac_address}:battery')
except redis.ResponseError:
    pass

try:
    redis_client.ts().create(f'{mac_address}:power')
except redis.ResponseError:
    pass

try:
    redis_client.ts().create(f'{mac_address}:plugged_seconds') #this ts stores how many seconds the power has been plugged in the last hour
except redis.ResponseError:
    pass

print(f'The mac address is: {mac_address}')

one_day_in_ms = 24 * 60 * 60 * 1000
one_hour_in_ms = 60 * 60 * 1000
one_minute_in_ms = 60*1000
redis_client.ts().alter(f'{mac_address}:battery', retention_msecs=one_day_in_ms)
redis_client.ts().alter(f'{mac_address}:power', retention_msecs=one_day_in_ms)
redis_client.ts().alter(f'{mac_address}:plugged_seconds', retention_msecs=30*one_day_in_ms)
redis_client.ts().createrule(f'{mac_address}:power', f'{mac_address}:plugged_seconds', 'sum', one_hour_in_ms)

while True:
    
    timestamp = time.time()
    timestamp_ms = int(timestamp * 1000)
    battery_level = psutil.sensors_battery().percent
    power_plugged = int(psutil.sensors_battery().power_plugged)
    formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

    redis_client.ts().add(f'{mac_address}:battery', timestamp_ms, battery_level)
    redis_client.ts().add(f'{mac_address}:power', timestamp_ms, power_plugged)
    
    
    time.sleep(1)  #acquisition time to 1 sec
    


