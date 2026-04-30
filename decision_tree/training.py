import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import random
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from labels import course_map
from decision_tree import (
    encode_gpa_input, 
    encode_course_input, 
    build_course_distribution, 
    sample_from_distribution
)

# TRAINING DATA

def load_gpa_data(path="gpa_data.json"):
    with open(path) as f:
        records = json.load(f)
    X = np.array([encode_gpa_input(r) for r in records])
    y = np.array([r["LABEL"] for r in records])
    return X, y

def load_course_data(path="course_data.json"):
    with open(path) as f:
        records = json.load(f)
    X        = np.array([encode_course_input(r) for r in records])
    y_labels = np.array([r["LABEL"] for r in records])
    # COURSES field: list of 3 expected courses per record (may be absent in old data)
    y_courses = np.array([r.get("COURSES", []) for r in records], dtype=object)
    return X, y_labels, y_courses

def _safe_stratify(y):
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) > 1 and counts.min() >= 2:
        return y
    return None

def _course_overlap_accuracy(predicted_courses_list, expected_courses_list):
    scores = []
    for predicted, expected in zip(predicted_courses_list, expected_courses_list):
        if expected is None or len(expected) == 0:  
            continue
        expected_set = set(expected)
        hits = sum(1 for c in predicted if c in expected_set)
        scores.append(hits / len(predicted) if predicted else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def train_and_validate(X, y, label="model", val_size=0.2, random_state=42, y_courses=None):
    strat = _safe_stratify(y)
    indices = np.arange(len(X))
    idx_train, idx_val = train_test_split(
        indices, test_size=val_size, random_state=random_state, stratify=strat
    )
    X_train, X_val   = X[idx_train], X[idx_val]
    y_train, y_val   = y[idx_train], y[idx_val]

    clf = RandomForestClassifier(n_estimators=50, random_state=random_state)
    clf.fit(X_train, y_train)

    # ADD HERE
    feature_names = ["GPA", "YEAR", "MAJOR", "CONCENTRATION"] if X.shape[1] == 4 else ["YEAR", "MAJOR", "CONCENTRATION", "LOAD", "COURSE_TYPE"]
    print(f"\n  -- Feature importances --")
    for name, imp in zip(feature_names, clf.feature_importances_):
        print(f"  {name}: {imp:.3f}")

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc   = accuracy_score(y_val,   clf.predict(X_val))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Train size: {len(X_train)}  |  Val size: {len(X_val)}")
    print(f"  -- Label accuracy --------------------------")
    print(f"  Train label accuracy : {train_acc:.3f}")
    print(f"  Val   label accuracy : {val_acc:.3f}")

    if y_courses is not None:
        y_courses_val = y_courses[idx_val]

        predicted_courses_list = []
        for x_row, expected_label in zip(X_val, y_val):
            pred_label_probs = {
                lbl: clf.predict_proba([x_row])[0][i]
                for i, lbl in enumerate(clf.classes_)
            }
            top_label = max(pred_label_probs, key=pred_label_probs.get)
            if pred_label_probs[top_label] > 0.6:
                available = course_map.get(top_label, {}).get("courses", [])
                sampled = random.sample(available, k=min(3, len(available))) if available else []
            else:
                dist    = build_course_distribution(pred_label_probs)
                sampled = sample_from_distribution(dist, k=3)
            predicted_courses_list.append(sampled)

        course_overlap = _course_overlap_accuracy(predicted_courses_list, y_courses_val)
        
        exact_matches = sum(
            1 for pred, exp in zip(predicted_courses_list, y_courses_val)
            if len(exp) > 0 and set(pred).issubset(set(exp))
        )
        exact_match_rate = exact_matches / len(y_courses_val) if len(y_courses_val) else 0

        print(f"\n  -- Course accuracy (probabilistic sampler) ---------")
        print(f"  Val course overlap accuracy : {course_overlap:.3f}")
        print(f"  Val exact-set match rate    : {exact_match_rate:.3f}")

    print(f"\n{classification_report(y_val, clf.predict(X_val), zero_division=0)}")
    print("="*60)

    return clf

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gpa_data_path = os.path.join(current_dir, "d_data", "gpa_data.json")
    course_data_path = os.path.join(current_dir, "d_data", "course_data.json")
    gpa_model_path = os.path.join(current_dir, "gpa_model.pkl")
    course_model_path = os.path.join(current_dir, "course_model.pkl")

    _X_gpa_fallback = np.array([[3.8, 3, 1, 0], [3.6, 2, 0, 8], [2.4, 2, 0, 8], [3.9, 4, 1, 0],
                                [2.8, 3, 1, 1], [3.2, 2, 1, 1], [3.7, 3, 1, 0], [2.6, 1, 0, 8]])
    _y_gpa_fallback = np.array(["ACCELERATE_TO_OPPORTUNITIES", "SUSTAIN_AND_OPTIMIZE", 
                                "FOUNDATIONAL_HABIT_BUILDING", "ACCELERATE_TO_OPPORTUNITIES",
                                "TARGETED_COURSE_RECOVERY", "SUSTAIN_AND_OPTIMIZE",
                                "ACCELERATE_TO_OPPORTUNITIES", "FOUNDATIONAL_HABIT_BUILDING"])

    _X_course_fallback = np.array([[1, 0, 8, 1, 0], [2, 1, 0, 1, 0], [3, 1, 0, 2, 0], [4, 1, 0, 2, 0],
                                   [2, 0, 8, 0, 1], [3, 1, 1, 1, 0], [3, 1, 2, 1, 0], [3, 1, 3, 1, 0]])
    _y_course_fallback = np.array(["FOLLOW_CORE_SEQUENCE", "CONCENTRATION_AI", "CHALLENGE_HEAVY_SEMESTER", 
                                   "SENIOR_DESIGN_MODE", "REBALANCE_LOAD", "CONCENTRATION_SOFTWARE",
                                   "CONCENTRATION_SECURITY", "CONCENTRATION_SYSTEMS"])

    print("=== STARTING TRAINING PROCESS ===")

    # --- GPA MODEL ---
    if os.path.exists(gpa_data_path):
        print(f"[INFO] Training new GPA model from {gpa_data_path}")
        X_gpa, y_gpa = load_gpa_data(gpa_data_path)
        gpa_model = train_and_validate(X_gpa, y_gpa, label="GPA MODEL")
    else:
        print(f"[INFO] {gpa_data_path} not found — using fallback data (prototype mode)")
        gpa_model = RandomForestClassifier(n_estimators=50, random_state=42)
        gpa_model.fit(_X_gpa_fallback, _y_gpa_fallback)
        
    with open(gpa_model_path, "wb") as f:
        pickle.dump(gpa_model, f)
    print(f"[INFO] Saved trained GPA model to {gpa_model_path}\n")

    # --- COURSE MODEL ---
    if os.path.exists(course_data_path):
        print(f"[INFO] Training new COURSE model from {course_data_path}")
        X_course, y_course, y_courses = load_course_data(course_data_path)
        course_model = train_and_validate(X_course, y_course, label="COURSE MODEL", y_courses=y_courses)
    else:
        print(f"[INFO] {course_data_path} not found — using fallback data (prototype mode)")
        course_model = RandomForestClassifier(n_estimators=50, random_state=42)
        course_model.fit(_X_course_fallback, _y_course_fallback)
        
    with open(course_model_path, "wb") as f:
        pickle.dump(course_model, f)
    print(f"[INFO] Saved trained COURSE model to {course_model_path}")

    print("=== TRAINING COMPLETE ===")
