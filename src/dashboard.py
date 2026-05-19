import streamlit as st
from src.monitoring import load_logs

def show_dashboard():

    st.subheader("Monitoring Dashboard")

    logs = load_logs()

    if not logs:
        st.warning("No prediction logs available")
        return

    # 🔹 total predictions
    total_predictions = sum(log["n_samples"] for log in logs)
    st.metric("Total Predictions", total_predictions)

    # 🔹 model usage
    model_counts = {}
    for log in logs:
        model = log["model"]
        model_counts[model] = model_counts.get(model, 0) + 1

    st.write("### Model Usage")
    st.write(model_counts)

    # 🔹 prediction distribution
    all_preds = []
    for log in logs:
        all_preds.extend(log["predictions"])

    st.write("### Prediction Distribution")
    st.write({
        "unique_predictions": list(set(all_preds)),
        "total_predictions": len(all_preds)
    })

    # 🔹 recent logs
    st.write("### Recent Activity")
    for log in logs[-5:]:
        st.write(f"{log['model']} (v{log['version']}) → {log['predictions']}")