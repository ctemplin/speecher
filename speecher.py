import os
import sys
import deepspeech as ds
import numpy as np
import pyaudio as pa

def main():
    model = ds.Model('../../deepspeech-0.7.4-models.pbmm')
    model.enableExternalScorer('../../deepspeech-0.7.4-models.scorer')

    # instantiate PyAudio (1)
    p = pa.PyAudio()

    print(p.get_default_input_device_info())

    stream = p.open(format=pa.paInt16,
                    channels=1,
                    input=True,
                    input_device_index=5,
                    rate=16000)

    stream.start_stream()

    rr = stream.read(100000)
    rr = np.frombuffer(rr, np.int16)
    # print(rr)

    ttt = model.stt(rr)
    print(ttt)

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    p.terminate()

    return

if __name__ == "__main__":
    main()

