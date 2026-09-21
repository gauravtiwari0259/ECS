import streamlit as st
import pandas as pd
import time
import os


st.set_page_config(page_title="AI Microgrid Dashboard", layout="wide")
st.title("⚡ AI-Assisted Smart Microgrid Live Telemetry")


placeholder = st.empty()


while True:
    if os.path.exists("live_microgrid_log.csv"):
        try:

            df = pd.read_csv("live_microgrid_log.csv")

            if not df.empty:
                latest = df.iloc[-1]


                with placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Solar Voltage", f"{latest['solar_v']:.2f} V")
                    col2.metric("Battery Voltage", f"{latest['batt_v']:.2f} V")
                    col3.metric("Load Demand", f"{latest['load_i']:.2f} A")

                    # Highlight the AI's action
                    action_color = "green" if latest['relay_action'] == "SOLAR_MODE" else "orange" if latest['relay_action'] == "BATTERY_MODE" else "red"
                    col4.markdown(f"### AI State: <span style='color:{action_color}'>{latest['relay_action']}</span>",
                                  unsafe_allow_html=True)

                    st.divider()


                    st.subheader("Real-Time Voltage Transients")
                    chart_data = df.set_index('timestamp')[['solar_v', 'batt_v']]
                    st.line_chart(chart_data, color=["#FFC107", "#00BCD4"])

            else:
                with placeholder.container():
                    st.info("Log file created, waiting for first ESP32 transmission...")

        except Exception as e:
            with placeholder.container():
                st.warning(f"Reading data... (Error: {e})")
    else:
        with placeholder.container():
            st.warning("⏳ Waiting for ESP32 live telemetry data. Ensure FastAPI server is running.")

    time.sleep(2)