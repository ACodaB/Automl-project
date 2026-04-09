import os
import joblib
import json

REGISTRY_PATH = "model_registry"


def get_next_version(model_name):
    model_dir = os.path.join(REGISTRY_PATH, model_name)

    if not os.path.exists(model_dir):
        return 1

    versions = [
        int(f.split("_v")[-1].split(".")[0])
        for f in os.listdir(model_dir)
        if f.endswith(".pkl")
    ]

    return max(versions) + 1 if versions else 1


def save_model(model, model_name, metadata):

    model_dir = os.path.join(REGISTRY_PATH, model_name)
    os.makedirs(model_dir, exist_ok=True)

    version = get_next_version(model_name)

    model_path = os.path.join(model_dir, f"{model_name}_v{version}.pkl")
    meta_path = os.path.join(model_dir, f"{model_name}_v{version}.json")

    # save model
    joblib.dump(model, model_path)

    #  save metadata
    metadata["version"] = version
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return model_path, version

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy types
        return obj.item()
    else:
        return obj
    
def load_latest_model(model_name):
    model_dir = os.path.join("model_registry", model_name)

    if not os.path.exists(model_dir):
        return None, None

    versions = [
        int(f.split("_v")[-1].split(".")[0])
        for f in os.listdir(model_dir)
        if f.endswith(".pkl")
    ]

    if not versions:
        return None, None

    latest_version = max(versions)

    model_path = os.path.join(
        model_dir,
        f"{model_name}_v{latest_version}.pkl"
    )

    model = joblib.load(model_path)

    return model, latest_version

def load_model_by_dataset(model_name, dataset_id):
    
    model_dir = os.path.join("model_registry", model_name)

    if not os.path.exists(model_dir):
        return None, None

    best_version = None
    best_path = None

    for file in os.listdir(model_dir):
        if file.endswith(".json"):

            version = int(file.split("_v")[-1].split(".")[0])
            meta_path = os.path.join(model_dir, file)

            with open(meta_path, "r") as f:
                metadata = json.load(f)

            if metadata.get("dataset_id") == dataset_id:
                if best_version is None or version > best_version:
                    best_version = version
                    best_path = os.path.join(
                        model_dir,
                        f"{model_name}_v{version}.pkl"
                    )

    if best_path is None:
        return None, None

    model = joblib.load(best_path)

    return model, best_version

def load_model_metadata(model_name, version):

    model_dir = os.path.join("model_registry", model_name)
    meta_path = os.path.join(model_dir, f"{model_name}_v{version}.json")

    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r") as f:
        return json.load(f)
    
import os

def get_registered_models():

    if not os.path.exists(REGISTRY_PATH):
        return []

    return [
        d for d in os.listdir(REGISTRY_PATH)
        if os.path.isdir(os.path.join(REGISTRY_PATH, d))
    ]
    
