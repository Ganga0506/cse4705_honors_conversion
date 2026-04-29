import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. COURSE MAP (probabilistic)
# -----------------------------
course_map = {
    "CAREER_TRACK_AI": {
        "courses": ["CSE4705", "CSE4820", "CSE4830", "CSE3500", "CSE2600"],
        "weights": [0.3, 0.25, 0.2, 0.15, 0.1]
    },
    "BALANCED_SEMESTER": {
        "courses": ["CSE2050", "CSE2600", "CSE2301", "CSE2102", "CSE2550"],
        "weights": [0.25, 0.25, 0.2, 0.15, 0.15]
    },
    "CHALLENGE_HEAVY_SEMESTER": {
        "courses": ["CSE3500", "CSE4300", "CSE4705", "CSE3666"],
        "weights": [0.3, 0.25, 0.25, 0.2]
    }
}

# -----------------------------
# 2. ENCODING HELPERS
# -----------------------------
course_load_map = {"Light": 0, "Medium": 1, "Hard": 2}

def encode_input(user):
    return [
        user["GPA"],
        user["YEAR"],
        course_load_map[user["COURSE_LOAD"]]
    ]

# -----------------------------
# 3. TRAINING DATA (dummy for now)
# Replace with better synthetic data later
# -----------------------------
X = np.array([
    [3.8, 3, 2],
    [3.6, 2, 1],
    [2.4, 2, 0],
    [3.9, 4, 2],
    [2.8, 3, 1],
    [3.2, 2, 1],
    [3.7, 3, 2],
    [2.6, 1, 0]
])

y = np.array([
    "CAREER_TRACK_AI",
    "BALANCED_SEMESTER",
    "BALANCED_SEMESTER",
    "CHALLENGE_HEAVY_SEMESTER",
    "BALANCED_SEMESTER",
    "BALANCED_SEMESTER",
    "CAREER_TRACK_AI",
    "BALANCED_SEMESTER"
])

# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# -----------------------------
# 5. PREDICT LABEL PROBABILITIES
# -----------------------------
def predict_label_distribution(user):
    encoded = encode_input(user)
    probs = model.predict_proba([encoded])[0]
    classes = model.classes_
    return dict(zip(classes, probs))

# -----------------------------
# 6. BUILD COURSE DISTRIBUTION
# -----------------------------
def build_course_distribution(label_probs):
    combined = {}

    for label, prob in label_probs.items():
        if label not in course_map:
            continue

        courses = course_map[label]["courses"]
        weights = course_map[label]["weights"]

        for c, w in zip(courses, weights):
            if c not in combined:
                combined[c] = 0
            combined[c] += prob * w

    return combined

# -----------------------------
# 7. FILTER COURSES (basic rules)
# -----------------------------
def filter_courses(course_dist, user):
    filtered = {}

    for course, score in course_dist.items():
        # Example rule: avoid 4000-level for Year 1
        if user["YEAR"] == 1 and course.startswith("CSE4"):
            continue

        filtered[course] = score

    return filtered

# -----------------------------
# 8. SAMPLE COURSES (weighted)
# -----------------------------
def sample_courses(course_dist, k_range=(2,3), temperature=0.7):
    courses = list(course_dist.keys())
    weights = np.array(list(course_dist.values()))

    # avoid division by zero
    if weights.sum() == 0:
        return []

    # temperature scaling
    weights = weights ** (1 / temperature)
    weights = weights / weights.sum()

    k = random.randint(*k_range)

    selected = list(np.random.choice(courses, size=k, replace=False, p=weights))
    return selected

# -----------------------------
# 9. FULL PIPELINE
# -----------------------------
def recommend_courses(user):
    label_probs = predict_label_distribution(user)

    course_dist = build_course_distribution(label_probs)

    filtered_dist = filter_courses(course_dist, user)

    recommendations = sample_courses(filtered_dist)

    return {
        "label_probabilities": label_probs,
        "course_distribution": filtered_dist,
        "recommended_courses": recommendations
    }

# -----------------------------
# 10. TEST RUN
# -----------------------------
if __name__ == "__main__":
    user_input = {
        "GPA": 3.8,
        "YEAR": 3,
        "COURSE_LOAD": "Hard"
    }

    result = recommend_courses(user_input)

    print("\nLabel Probabilities:")
    for k, v in result["label_probabilities"].items():
        print(f"{k}: {round(v, 3)}")

    print("\nRecommended Courses:")
    print(result["recommended_courses"])