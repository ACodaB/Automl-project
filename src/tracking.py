import json
import datetime
def log_experiment(run_data,filepath="experiments.json"):
    run_data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(filepath, "a") as f:
            f.write(json.dumps(run_data) + "\n")
    except Exception as e:
        print(f"Logging failed: {e}")