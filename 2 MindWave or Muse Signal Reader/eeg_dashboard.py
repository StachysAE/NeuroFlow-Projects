# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import time

# -------------------------------
# 1. Load EEG data (Muse CSV file)
# -------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv('eeg.csv')  # Replace with your Muse CSV file
    return df

df = load_data()

# Check available columns
st.write("Loaded EEG data columns:", df.columns.tolist())

# We'll use Alpha and Theta bands from TP9 sensor (example)
alpha_col = 'Alpha_TP9'
theta_col = 'Theta_TP9'

# -------------------------------
# 2. Initialize placeholders
# -------------------------------

st.title('🧠 Real-time EEG Attention & Relaxation Dashboard')

chart_placeholder = st.empty()
bar_placeholder = st.empty()

# Initialize history
attention_history = []
relaxation_history = []

# -------------------------------
# 3. Define functions
# -------------------------------

def compute_attention_relaxation(alpha_signal, theta_signal):
    alpha_mean = alpha_signal.mean()
    theta_mean = theta_signal.mean()

    # Skip if too small (prevent division explosion)
    if alpha_mean < 0.5 or theta_mean < 0.5:
        return None, None

    attention_score = theta_mean / (alpha_mean + 1e-6)
    relaxation_score = alpha_mean / (theta_mean + 1e-6)

    return attention_score, relaxation_score

# -------------------------------
# 4. Streaming Loop
# -------------------------------

window_size = 256  # 1 second window (assuming 256Hz sampling rate)

for start in range(0, len(df) - window_size, window_size):
    eeg_window = df.iloc[start : start + window_size]

    alpha_signal = eeg_window[alpha_col].values
    theta_signal = eeg_window[theta_col].values

    # Skip bad windows
    if (np.isnan(alpha_signal).any() or np.isnan(theta_signal).any() or
        np.all(alpha_signal == 0) or np.all(theta_signal == 0)):
        continue

    # Calculate scores
    attention_score, relaxation_score = compute_attention_relaxation(alpha_signal, theta_signal)

    if attention_score is None or relaxation_score is None:
        continue  # Skip bad window

    # Append to history
    attention_history.append(attention_score)
    relaxation_history.append(relaxation_score)

    # Plot Line Chart
    history_df = pd.DataFrame({
        'Attention': attention_history,
        'Relaxation': relaxation_history
    })

    chart_placeholder.line_chart(history_df)

    # Plot Current Bar Chart
    current_df = pd.DataFrame({
        'State': ['Attention', 'Relaxation'],
        'Score': [attention_score, relaxation_score]
    })

    bar_placeholder.bar_chart(current_df.set_index('State'))

    # Small delay to simulate real-time
    time.sleep(0.2)  # Faster update for smoothness
