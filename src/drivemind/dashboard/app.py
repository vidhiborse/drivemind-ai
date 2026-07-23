"""
DriveMind AI — Streamlit Dashboard
Consumes the FastAPI backend (must be running separately at localhost:8000)
to show live session status and historical trip data.

Run with: streamlit run src/drivemind/dashboard/app.py
"""

import streamlit as st
import requests
import pandas as pd
import time

API_BASE = "http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="DriveMind AI Dashboard", layout="wide")

st.title("🚗 DriveMind AI — Driver Safety Dashboard")

# Initialize session state to remember the active trip_id across reruns
if "active_trip_id" not in st.session_state:
    st.session_state.active_trip_id = None

RISK_COLORS = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "CRITICAL": "🔴",
}

# --- Section 1: Live Session Control ---
st.header("Live Session")

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    if st.button("▶️ Start Session", disabled=st.session_state.active_trip_id is not None):
        try:
            response = requests.post(f"{API_BASE}/session/start", timeout=5)
            data = response.json()
            st.session_state.active_trip_id = data["trip_id"]
            st.success(f"Session started (Trip ID: {data['trip_id']})")
        except Exception as e:
            st.error(f"Could not start session. Is the API running? ({e})")

with col2:
    if st.button("⏹️ Stop Session", disabled=st.session_state.active_trip_id is None):
        try:
            requests.post(f"{API_BASE}/session/{st.session_state.active_trip_id}/stop", timeout=5)
            st.session_state.active_trip_id = None
            st.info("Session stopped.")
        except Exception as e:
            st.error(f"Could not stop session. ({e})")

# --- Section 2: Live Status Display ---
# Using st.empty() as a placeholder ensures each refresh CLEANLY REPLACES the
# previous content instead of stacking new elements on top of old ones —
# this is what prevents duplicate/overlapping "Reasons" text on each rerun.
status_placeholder = st.empty()

if st.session_state.active_trip_id is not None:
    try:
        response = requests.get(
            f"{API_BASE}/session/{st.session_state.active_trip_id}/status", timeout=5
        )
        status = response.json()

        decision = status.get("latest_decision")
        fatigue = status.get("latest_fatigue")

        with status_placeholder.container():
            st.subheader(f"Live Status — Trip {status['trip_id']}")

            status_col1, status_col2, status_col3 = st.columns(3)

            with status_col1:
                if decision:
                    risk = decision["risk_level"]
                    st.metric("Risk Level", f"{RISK_COLORS.get(risk, '')} {risk}")
                else:
                    st.metric("Risk Level", "Initializing...")

            with status_col2:
                if decision and decision.get("identity", {}).get("driver_identified"):
                    driver_name = decision["identity"]["driver_name"]
                    st.metric("Driver", driver_name.upper())
                else:
                    st.metric("Driver", "Detecting...")

            with status_col3:
                if fatigue and fatigue.get("fatigue_probability_10min") is not None:
                    pct = int(fatigue["fatigue_probability_10min"] * 100)
                    st.metric("Fatigue (10min)", f"{pct}% ({fatigue['trend']})")
                else:
                    st.metric("Fatigue (10min)", "Calculating...")

            if decision:
                st.write("**Reasons:**")
                for reason in decision["reasons"]:
                    st.write(f"- {reason}")

        # Auto-refresh every 2 seconds while a session is active
        time.sleep(2)
        st.rerun()

    except Exception as e:
        with status_placeholder.container():
            st.error(f"Could not fetch live status. ({e})")

st.divider()

# --- Section 3: Trip History ---
st.header("Trip History")

try:
    response = requests.get(f"{API_BASE}/trips/", timeout=5)
    trips = response.json()

    if trips:
        df = pd.DataFrame(trips)
        st.dataframe(df, use_container_width=True)

        # --- Section 4: Risk History Chart for a selected trip ---
        st.subheader("Risk History for a Trip")
        trip_ids = [t["id"] for t in trips]
        selected_trip = st.selectbox("Select a trip to view risk history", trip_ids)

        if selected_trip:
            risk_response = requests.get(f"{API_BASE}/trips/{selected_trip}/risk-history", timeout=5)
            risk_data = risk_response.json()

            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                risk_level_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
                risk_df["risk_numeric"] = risk_df["risk_level"].map(risk_level_map)
                st.line_chart(risk_df.set_index("timestamp")["risk_numeric"])
            else:
                st.write("No risk history logged for this trip.")
    else:
        st.write("No trips recorded yet. Start a session above.")

except Exception as e:
    st.error(f"Could not fetch trip history. Is the API running? ({e})")