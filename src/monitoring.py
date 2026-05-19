import json
import datetime
import os

LOG_FILE = "prediction_logs.json"

def log_prediction(model_name, version, input_data, predictions):

    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "version": version,
        "n_samples": len(predictions),
        "predictions": predictions.tolist()
    }

    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Logging failed: {e}")

def load_logs(filepath="prediction_logs.json"):

    if not os.path.exists(filepath):
        return []

    logs = []

    with open(filepath, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except:
                continue

    return logs