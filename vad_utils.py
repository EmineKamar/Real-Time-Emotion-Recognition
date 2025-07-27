import collections
import sys
import signal
import webrtcvad
import pyaudio
import struct
import time
# vad_utils.py
import pyaudio
import numpy as np

CHUNK = 16000  # 1 saniye
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def get_audio_chunk():
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

def close_stream():
    stream.stop_stream()
    stream.close()
    p.terminate()

class VADAudio:
    def __init__(
        self,
        aggressiveness=3,
        input_rate=16000,
        frame_duration=30,
        padding_duration=1500,
        device=None,
    ):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.input_rate = input_rate
        self.frame_duration = frame_duration  # in ms
        self.frame_size = int(input_rate * frame_duration / 1000)  # in samples
        self.bytes_per_sample = 2
        self.frame_bytes = self.frame_size * self.bytes_per_sample
        self.padding_ms = padding_duration
        self.num_padding_frames = int(padding_duration / frame_duration)
        self.device = device

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=input_rate,
            input=True,
            frames_per_buffer=self.frame_size,
            input_device_index=device,
        )

        self.running = True
        signal.signal(signal.SIGINT, self.stop)

    def stop(self, *args):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def frame_generator(self):
        while self.running:
            data = self.stream.read(self.frame_size, exception_on_overflow=False)
            yield data

    def vad_collector(self):
        ring_buffer = collections.deque(maxlen=self.num_padding_frames)
        triggered = False
        voiced_frames = []

        for frame in self.frame_generator():
            is_speech = self.vad.is_speech(frame, self.input_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    for f, _ in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b"".join(voiced_frames)
                    ring_buffer.clear()
                    voiced_frames = []

        if voiced_frames:
            yield b"".join(voiced_frames)

