# Delta-Modulation
Aim:

To implement Delta Modulation (DM) for encoding a continuous-time signal into a digital format, and then reconstruct it using Delta Demodulation with a low-pass filter.

Tools required:

Python software(Version 3.6) with,

-> Numpy Library (for numerical operations)

-> Matplotlib Library (for signal visualization)

-> Scipy Library (for signal processing and filtering)

Program:


import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

fs = 10000  # Sampling frequency

f = 10  # Signal frequency

T = 1  # Duration in seconds

delta = 0.1  # Step size

t = np.arange(0, T, 1/fs)

message_signal = np.sin(2 * np.pi * f * t)  # Sine wave as input signal

encoded_signal = []

dm_output = [0]  # Initial value of the modulated signal

prev_sample = 0

for sample in message_signal:

    if sample > prev_sample:
    
        encoded_signal.append(1)
        
        dm_output.append(prev_sample + delta)
        
    else:
    
        encoded_signal.append(0)
    
        dm_output.append(prev_sample - delta)
    
    prev_sample = dm_output[-1]

demodulated_signal = [0]

for bit in encoded_signal:

    if bit == 1:
    
        demodulated_signal.append(demodulated_signal[-1] + delta)
    
    else:
        
        demodulated_signal.append(demodulated_signal[-1] - delta)

demodulated_signal = np.array(demodulated_signal)

def low_pass_filter(signal, cutoff_freq, fs, order=4):
    
    nyquist = 0.5 * fs
    
    normal_cutoff = cutoff_freq / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    return filtfilt(b, a, signal)

filtered_signal = low_pass_filter(demodulated_signal, cutoff_freq=20, fs=fs)

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)

plt.plot(t, message_signal, label='Original Signal', linewidth=1)

plt.legend()

plt.grid()

plt.subplot(3, 1, 2)

plt.step(t, dm_output[:-1], label='Delta Modulated Signal', where='mid')

plt.legend()

plt.grid()

plt.subplot(3, 1, 3)

plt.plot(t, filtered_signal[:-1], label='Demodulated & Filtered Signal', linestyle='dotted', linewidth=1, color='r')

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()

Output Waveform:

![Screenshot 2025-03-28 104726](https://github.com/user-attachments/assets/94348c81-f0f5-4fc4-a883-5dfad62c7d86)

![Screenshot 2025-03-28 104753](https://github.com/user-attachments/assets/272ce9cd-692b-4f18-835a-9647fee8b2cd)

![Screenshot 2025-03-28 104820](https://github.com/user-attachments/assets/d5c6ec05-d477-4d49-9975-9e1ca8053f1f)



Result:

Delta Modulation successfully converts a continuous sine wave into a 1-bit digital signal. The demodulated signal, after applying a low-pass filter, closely resembles the original signal but may have quantization errors and slope overload distortion. Proper step size selection (Δ) improves accuracy.
