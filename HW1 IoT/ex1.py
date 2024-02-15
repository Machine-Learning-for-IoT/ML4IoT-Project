import argparse
import os
import sounddevice as sd
from time import time
from scipy.io.wavfile import write
import tensorflow as tf

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


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
optimal_vad = VAD(16000, 0.032, 20, 50, 450, -95, 0.1)


buffer_shape = (16000,)
audio_buffer = tf.zeros(shape=buffer_shape, dtype=tf.float32)


def callback(indata, frames, callback_time, status):
    """This is called (from a separate thread) for each audio block."""
    global audio_buffer
    
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
        timestamp = time()
        try:
            write(f'{timestamp}.wav', 16000, (tf.cast(audio_buffer, tf.int16)*tf.int16.max).numpy())
            filesize_in_bytes = os.path.getsize(f'{timestamp}.wav')
            filesize_in_kb = filesize_in_bytes / 1024
            print(f'Size: {filesize_in_kb:.2f}KB')
        except Exception as e:
            print(f'Error writing WAV file: {e}')
    else:
        print('Silence')


blocksize = 8000 #it has been set equal to 8000 due to 0.5s @ 16 kHz
with sd.InputStream(device=args.device, channels=1, dtype='int16', samplerate=16000, blocksize=blocksize, callback=callback):
    while True:
        key = input()
        if key in ('q', 'Q'):
            print('Stop recording.')
            break

