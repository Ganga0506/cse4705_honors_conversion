import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import random
import pickle
import numpy as np
from labels import course_map, GPA_RESPONSES, course_response

# ENCODING HELPERS
major_map         = {"CS": 0, "CSE": 1, "DSE": 2}
concentration_map = {
    "Artificial Intelligence": 0,
    "Software Design and Development": 1,
    "Cybersecurity": 2,
    "Systems and Networks": 3,
    "Software Design for Mobile Computing": 4,
    "Naval Science and Technology": 5,
    "Algorithms and Theory": 6,
    "Bioinformatics": 7,
    "None": 8,
}
load_map        = {"Light": 0, "Medium": 1, "Hard": 2}
course_type_map = {"Core": 0, "Elective": 1}

def encode_gpa_input(user):
    if user["MAJOR"] not in major_map:
        raise ValueError(f"Invalid MAJOR '{user['MAJOR']}'. Expected one of: {list(major_map.keys())}")
    return [
        user["GPA"],
        user["YEAR"],
        major_map[user["MAJOR"]],
        concentration_map.get(user["CONCENTRATION"], 8),
    ]

def encode_course_input(user):
    if user["MAJOR"] not in major_map:
        raise ValueError(f"Invalid MAJOR '{user['MAJOR']}'. Expected one of: {list(major_map.keys())}")
    return [
        user["YEAR"],
        major_map[user["MAJOR"]],
        concentration_map.get(user["CONCENTRATION"], 8),
        load_map.get(user["LOAD"], 1),
        course_type_map.get(user["COURSE_TYPE"], 0),
    ]

def build_course_distribution(label_probs):
    combined = {}
    for label, prob in label_probs.items():
        if label not in course_map:
            continue
        courses = course_map[label]["courses"]
        weights = course_map[label]["weights"]
        for c, w in zip(courses, weights):
            combined[c] = combined.get(c, 0) + prob * w
    return combined

def sample_from_distribution(course_dist, k=3):
    courses = list(course_dist.keys())
    weights = np.array(list(course_dist.values()))

    if weights.sum() == 0:
        return random.sample(courses, k=min(k, len(courses)))

    weights = weights / weights.sum()
    return list(np.random.choice(courses, size=min(k, len(courses)), replace=False, p=weights))

# LOAD MODELS
current_dir = os.path.dirname(os.path.abspath(__file__))
gpa_model_path = os.path.join(current_dir, "gpa_model.pkl")
course_model_path = os.path.join(current_dir, "course_model.pkl")

gpa_model = None
course_model = None

if os.path.exists(gpa_model_path):
    with open(gpa_model_path, "rb") as f:
        gpa_model = pickle.load(f)
else:
    print(f"[WARNING] {gpa_model_path} not found. Please run decision_tree/training.py first!")

if os.path.exists(course_model_path):
    with open(course_model_path, "rb") as f:
        course_model = pickle.load(f)
else:
    print(f"[WARNING] {course_model_path} not found. Please run decision_tree/training.py first!")


# PREDICT + SAMPLE

def predict_gpa_label(user):
    probs = gpa_model.predict_proba([encode_gpa_input(user)])[0]
    return {label: probs[i] for i, label in enumerate(gpa_model.classes_)}

def predict_course_label(user):
    probs = course_model.predict_proba([encode_course_input(user)])[0]
    return {label: probs[i] for i, label in enumerate(course_model.classes_)}

# PIPELINES

def run_gpa_help(user):
    label_probs = predict_gpa_label(user)
    top_label   = max(label_probs, key=label_probs.get)
    return GPA_RESPONSES[top_label]

def run_course_selection(user):
    label_probs = predict_course_label(user)
    top_label   = max(label_probs, key=label_probs.get)

    if label_probs[top_label] > 0.6:
        available = course_map.get(top_label, {}).get("courses", [])
        courses = random.sample(available, k=min(3, len(available))) if available else []
    else:
        course_dist = build_course_distribution(label_probs)
        courses     = sample_from_distribution(course_dist, k=3)

    return course_response(top_label, courses)

# ENTRY POINT

def recommend(inputs):
    if gpa_model is None or course_model is None:
        raise RuntimeError("Models are not loaded. Please run 'python decision_tree/training.py' to train the models.")
        
    topic = inputs[0].lower()
    if topic == "gpa_help":
        _, gpa, major, concentration, year = inputs
        return run_gpa_help({"GPA": gpa, "MAJOR": major, "CONCENTRATION": concentration, "YEAR": year})
    elif topic == "course_selection":
        _, major, concentration, load, course_type, year = inputs
        return run_course_selection({"MAJOR": major, "CONCENTRATION": concentration, "LOAD": load, "COURSE_TYPE": course_type, "YEAR": year})
    else:
        raise ValueError(f"Unknown topic '{inputs[0]}'. Use 'gpa_help' or 'course_selection'.")