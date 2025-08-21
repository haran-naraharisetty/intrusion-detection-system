import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import io

st.title(" Intrusion Detection System (IDS) - Streamlit App")
st.markdown("Upload network traffic data to detect potential intrusions using an unsupervised machine learning model (Isolation Forest).")

#  Step 1: Simulate training data
@st.cache_data
def train_model():
    data = pd.DataFrame({
        'src_port': np.random.randint(1024, 65535, 1000),
        'dst_port': np.random.randint(20, 1024, 1000),
        'packet_size': np.random.normal(500, 50, 1000),
        'duration': np.random.exponential(scale=1.0, size=1000)
    })
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)
    return model, scaler

model, scaler = train_model()

#  Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file (must contain: src_port, dst_port, packet_size, duration)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader(" Uploaded Data")
        st.dataframe(df)

        required_cols = ['src_port', 'dst_port', 'packet_size', 'duration']
        if not all(col in df.columns for col in required_cols):
            st.error("CSV must contain columns: src_port, dst_port, packet_size, duration")
        else:
            X_input = df[required_cols]
            X_scaled_input = scaler.transform(X_input)
            predictions = model.predict(X_scaled_input)

            df['Prediction'] = [' Intrusion' if p == -1 else '✅ Normal' for p in predictions]

            st.subheader(" Prediction Results")
            st.dataframe(df)

            # Allow download of result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(" Download Results as CSV", data=csv, file_name="detection_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
