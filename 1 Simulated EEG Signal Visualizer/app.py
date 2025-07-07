# Save this as app.py and run with `streamlit run app.py`

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Simulate EEG signal
def generate_eeg(fs=250, duration=2):
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    delta = np.sin(2 * np.pi * 2 * t)
    theta = np.sin(2 * np.pi * 6 * t)
    alpha = np.sin(2 * np.pi * 10 * t)
    beta = np.sin(2 * np.pi * 20 * t)
    gamma = np.sin(2 * np.pi * 40 * t)
    noise = np.random.randn(len(t)) * 0.5
    eeg = delta + theta + alpha + beta + gamma + noise
    return t, eeg

# FFT Analysis
def plot_fft(eeg, fs):
    n = len(eeg)
    yf = fft(eeg)
    xf = fftfreq(n, 1 / fs)
    plt.plot(xf[:n // 2], np.abs(yf[:n // 2]))
    plt.title("EEG Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

# Streamlit UI
st.title("🧠 Simulated EEG Signal Visualizer")

fs = st.slider("Sampling Rate (Hz)", 100, 1000, 250)
duration = st.slider("Duration (seconds)", 1, 10, 2)

t, eeg = generate_eeg(fs, duration)

st.subheader("Raw EEG Signal")
fig1, ax1 = plt.subplots()
ax1.plot(t, eeg)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Simulated EEG")
st.pyplot(fig1)

st.subheader("Frequency Spectrum")
fig2, ax2 = plt.subplots()
plot_fft(eeg, fs)
st.pyplot(fig2)
