import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Load trained self-healing model
# --------------------------------------------------
model = load_model("self_healing_network_model.h5")

# --------------------------------------------------
# Simulate real-time network data
# --------------------------------------------------
def get_network_data():
    csv_traffic = np.random.randint(100, 1000)
    cicids_error = np.random.choice([0, 1])
    cicids_attack = np.random.choice([0, 1])
    return csv_traffic, cicids_error, cicids_attack

# --------------------------------------------------
# Preprocessing (MATCH MODEL INPUTS EXACTLY)
# Model Inputs:
#   Input 0 → CSV features → (None, 7)
#   Input 1 → CICIDS features → (None, 78)
# --------------------------------------------------
def preprocess_for_model(csv_traffic, cicids_data):
    # -------- CSV INPUT (7 FEATURES) --------
    csv_input = np.zeros((1, 7))
    csv_input[0, 0] = csv_traffic
    csv_input[0, 1:] = np.random.rand(6)

    # -------- CICIDS INPUT (78 FEATURES) --------
    cicids_input = np.random.rand(1, 78)

    # Embed logical signals into CICIDS vector
    cicids_input[0, 0] = cicids_data[0]   # error flag
    cicids_input[0, 1] = cicids_data[1]   # malicious activity flag

    return csv_input, cicids_input

# --------------------------------------------------
# AI Prediction
# --------------------------------------------------
def predict_healing_action(csv_traffic, cicids_data):
    csv_input, cicids_input = preprocess_for_model(csv_traffic, cicids_data)
    prediction = model.predict([csv_input, cicids_input], verbose=0)
    return float(prediction[0][0])

# --------------------------------------------------
# Visualization
# --------------------------------------------------
def create_mesh_graph(csv_traffic, cicids_data, healing_action):
    fig = go.Figure(data=[go.Mesh3d(
        x=[csv_traffic, csv_traffic],
        y=[cicids_data[0], cicids_data[1]],
        z=[0, healing_action],
        opacity=0.6
    )])

    fig.update_layout(
        title="Real-Time Self-Healing Network State",
        scene=dict(
            xaxis_title="Traffic Volume",
            yaxis_title="Errors / Attacks",
            zaxis_title="Healing Probability"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Self-Healing Network", layout="wide")
st.title("AI-Based Self-Healing Network Dashboard")

st.write("Live simulation of network conditions and autonomous healing decisions.")

container = st.empty()

# --------------------------------------------------
# Real-time Simulation Loop
# --------------------------------------------------
while True:
    with container.container():
        traffic, error, attack = get_network_data()

        st.subheader("📡 Network Metrics")
        col1, col2, col3 = st.columns(3)

        col1.metric("Traffic", traffic)
        col2.metric("Errors Detected", "Yes" if error else "No")
        col3.metric("Malicious Activity", "Yes" if attack else "No")

        healing = predict_healing_action(traffic, (error, attack))

        st.subheader("🛠 AI Decision Engine")
        if healing > 0.5:
            st.error("🚨 Self-Healing Initiated")
        else:
            st.success("✅ Network Stable")

        create_mesh_graph(traffic, (error, attack), healing)

    time.sleep(5)
    st.experimental_rerun()
