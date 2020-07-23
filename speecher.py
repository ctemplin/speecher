import os
import sys
import time
import deepspeech as ds
import numpy as np
import pyaudio as pa

model = ds.Model('../../deepspeech-0.7.4-models.pbmm')
model.enableExternalScorer('../../deepspeech-0.7.4-models.scorer')
dsStream = model.createStream()

def pa_callback(in_data,      # recorded data if input=True; else None
                frame_count,  # number of frames
                time_info,    # dictionary
                status_flags):# PaCallbackFlags

    rr = np.frombuffer(in_data, np.int16)
    dsStream.feedAudioContent(rr)
    ttt = dsStream.intermediateDecode()
    print(ttt)

    out_data = None
    flag = pa.paContinue
    return (out_data, flag)

def main():
    # instantiate PyAudio (1)
    p = pa.PyAudio()

    input_device_index = p.get_default_input_device_info()['index']

    stream = p.open(format=pa.paInt16,
                    channels=1,
                    input=True,
                    input_device_index=input_device_index,
                    rate=16000,
                    frames_per_buffer=2048*16,
                    stream_callback=pa_callback)

    stream.start_stream()

    try:
        while stream.is_active:
            time.sleep(5)
    except KeyboardInterrupt:
        print('interrupted!')
        stream.stop_stream()
        stream.close()
        p.terminate()
        dsStream.freeStream()
        return

if __name__ == "__main__":
    main()

