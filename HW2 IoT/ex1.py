import argparse
import os
import sounddevice as sd
import time
from datetime import datetime
from scipy.io.wavfile import write
import tensorflow as tf
import numpy as np
import psutil
import redis
import uuid


class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio, 
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        return spectrogram

class MelSpectrogram():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    ):
        self.spectrogram_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram
    
class MFCC():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        num_coefficients
    ):
        self.sampling_rate = sampling_rate
        self.frame_length_in_s = frame_length_in_s
        self.frame_step_in_s = frame_step_in_s
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        self.log_mel_spectogram_processor = MelSpectrogram(sampling_rate,frame_length_in_s,frame_step_in_s,num_mel_bins,lower_frequency,upper_frequency)

    def get_mfccs(self, audio):
        # TODO: Write your code here
        log_mel_spectrogram = self.log_mel_spectogram_processor.get_mel_spec(audio)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

        return mfccs


    def get_mfccs_and_label(self, audio, label):
        mfccs = self.get_mfccs(audio)

        return mfccs, label

class VAD():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        dbFSthres, 
        duration_thres
    ):
        self.sampling_rate = sampling_rate
        self.frame_length_in_s = frame_length_in_s
        self.mel_spec_processor = MelSpectrogram(
            sampling_rate, frame_length_in_s, frame_length_in_s, num_mel_bins, lower_frequency, upper_frequency
        )
        self.dbFSthres = dbFSthres
        self.duration_thres = duration_thres

    def is_silence(self, audio):
        log_mel_spec = self.mel_spec_processor.get_mel_spec(audio)
        dbFS = 20 * log_mel_spec
        energy = tf.math.reduce_mean(dbFS, axis=1)

        non_silence = energy > self.dbFSthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        non_silence_duration = (non_silence_frames + 1) * self.frame_length_in_s

        if non_silence_duration > self.duration_thres:
            return 0
        else:
            return 1




PREPROCESSING_ARGS = { 'sampling_rate': 16000,
    'frame_length_in_s': 0.032,
    'frame_step_in_s': 0.016,
    'num_mel_bins': 16, # Triet 32
    'lower_frequency': 20,
    'upper_frequency': 4000,
    'num_coefficients': 20 # Triet 13
}

mfcc_spec_processor = MFCC(**PREPROCESSING_ARGS)
interpreter = tf.lite.Interpreter(model_path="HW2_Team25/model25.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--host', type = str, default= '')
parser.add_argument('--port', type = int, default= 0)
parser.add_argument('--user', type = str, default='')
parser.add_argument('--password', type = str, default='')
args = parser.parse_args()
optimal_vad = VAD(16000, 0.032, 20, 50, 450, -95, 0.1)
buffer_shape = (16000,)
audio_buffer = tf.zeros(shape=buffer_shape, dtype=tf.float32)

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


print(f'The mac address is: {mac_address}')

one_day_in_ms = 24 * 60 * 60 * 1000
one_hour_in_ms = 60 * 60 * 1000
one_minute_in_ms = 60*1000
redis_client.ts().alter(f'{mac_address}:battery', retention_msecs=one_day_in_ms)
redis_client.ts().alter(f'{mac_address}:power', retention_msecs=one_day_in_ms)

counter = 0 #we are implementing a global counter for the callback function
# because the callback function it is called every 0.5 seconds and if the monitoring
# is active, we should monitor the battery level every 1 seconds, so every time the 
#callback function is invoked we increase the counter of 1, and if the counter is even
# we record the battery level (so every 1 seconds)
do_monitoring = False

def callback(indata, frames, callback_time, status):
    """This is called (from a separate thread) for each audio block."""
    global audio_buffer
    global do_monitoring
    global counter
    #print(f'Indata: {np.shape(indata)}')
    tf_indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    #print(f'Tf_indata: {tf_indata.shape}')
    squeezed = tf.squeeze(tf_indata)
    #print(f'Squeezed: {squeezed.shape}')
    normalized = squeezed / tf.int16.max
    #print(f'Normalized: {normalized.shape}')


    audio_buffer = tf.concat([audio_buffer, normalized], axis=0)


    # Remove the first 8000 elements
    audio_buffer = audio_buffer[8000:]
    
    silence = optimal_vad.is_silence(audio_buffer)
        
    if silence == 0:
        print('No Silence')
        timestamp = time.time()
        # try:
        #     write(f'{timestamp}.wav', 16000, (audio_buffer*tf.int16.max).numpy().astype(np.int16))
        #     filesize_in_bytes = os.path.getsize(f'{timestamp}.wav')
        #     filesize_in_kb = filesize_in_bytes / 1024
        #     print(f'Size: {filesize_in_kb:.2f}KB')
        # except Exception as e:
        #     print(f'Error writing WAV file: {e}')
        
        mfcc = mfcc_spec_processor.get_mfccs(audio_buffer)
        mfcc = tf.expand_dims(mfcc, 0)
        mfcc = tf.expand_dims(mfcc, -1)
        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if(output[0][0] >= 0.99):
            print("you said yes and the monitoring is active")
            do_monitoring = True
        elif(output[0][1] >= 0.99):
            print("you said no and the monitoring is not active")
            do_monitoring = False
        else:
            print("you said something else")
        
    else:
        print('Silence')
    counter += 1 
    if(do_monitoring == True and counter%2 == 0):
        timestamp = time.time()
        timestamp_ms = int(timestamp * 1000)
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)
        redis_client.ts().add(f'{mac_address}:battery', timestamp_ms, battery_level)
        redis_client.ts().add(f'{mac_address}:power', timestamp_ms, power_plugged)
        counter = 0


blocksize = 8000 #it has been set equal to 8000 due to 0.5s @ 16 kHz
with sd.InputStream(device=args.device, channels=1, dtype='int16', samplerate=16000, blocksize=blocksize, callback=callback):
    while True:
        key = input()
        
            
        




    