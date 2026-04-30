import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. COURSE MAP
course_map = {

    "FOLLOW_CORE_SEQUENCE": {
        "courses": ["CSE1729", "CSE2050", "CSE2301", "CSE2500", "CSE3500", "CSE3666"],
        "weights": [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
    },

    "REBALANCE_LOAD": {
        "courses": ["CSE2102", "CSE2600", "CSE3000", "CSE3160", "CSE2550", "CSE3200", "CSE3250"],
        "weights": [0.20, 0.20, 0.15, 0.15, 0.15, 0.10, 0.05]
    },

    "CONCENTRATION_AI": {
        "courses": ["CSE4705", "CSE4820", "CSE4830", "CSE4502", "CSE3500", "CSE4704", "CSE4701"],
        "weights": [0.25, 0.25, 0.20, 0.10, 0.08, 0.07, 0.05]
    },

    "CONCENTRATION_SOFTWARE": {
        "courses": ["CSE4102", "CSE4100", "CSE3100", "CSE4300", "CSE4701", "CSE3160", "CSE2102"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "CONCENTRATION_SECURITY": {
        "courses": ["CSE4400", "CSE4402", "CSE3400", "CSE3140", "CSE4702", "CSE4412", "CSE3300"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "CONCENTRATION_SYSTEMS": {
        "courses": ["CSE4300", "CSE4302", "CSE3300", "CSE3100", "CSE3666", "CSE4709", "CSE3504"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "CONCENTRATION_MOBILE": {
        "courses": ["CSE4939W", "CSE3200", "CSE3250", "CSE4100", "CSE3150", "CSE4102", "CSE3160"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "CONCENTRATION_NAVAL": {
        "courses": ["CSE3300", "CSE3666", "CSE4300", "CSE4302", "CSE3504", "CSE3140", "CSE4709"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "CONCENTRATION_ALGORITHMS": {
        "courses": ["CSE3500", "CSE4502", "CSE4701", "CSE3502", "CSE4705", "CSE3000", "CSE4830"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "CONCENTRATION_BIOINFORMATICS": {
        "courses": ["CSE3802", "CSE3800", "CSE3810", "CSE4502", "CSE4820", "CSE4830", "CSE4701"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "MAJOR_DATA_SCIENCE": {
        "courses": ["CSE4502", "CSE4701", "CSE2600", "CSE4820", "CSE3800", "CSE3810", "CSE3802"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    # ALL majors — CSE core sequence + full math prereq sequence across all levels
    # (linear algebra, diff eq, multivariable calc, probability all gate real CSE work)
    "PREREQ_CATCH_UP": {
        "courses": [
            "CSE1729", "CSE2050", "CSE2500", "CSE3500",
            "MATH1131Q", "MATH1132Q", "MATH2110Q", "MATH2210Q", "MATH2410Q",
            "MATH3160", "STAT3025Q", "STAT3345Q",
        ],
        "weights": [0.15, 0.13, 0.10, 0.08, 0.12, 0.10, 0.09, 0.09, 0.07, 0.04, 0.02, 0.01]
    },

    # CSE (BSE) major only — engineering-specific courses CS/DSE students don't take
    # includes CSE courses only BSE students need + ECE2001 + physics sequence
    "ENGINEERING_CORE": {
        "courses": [
            "CSE2301", "CSE3504", "CSE3666", "CSE3302",
            "ECE2001", "PHYS1501Q", "PHYS1502Q",
            "MATH2210Q", "MATH2410Q", "MATH3160", "STAT3345Q",
        ],
        "weights": [0.18, 0.15, 0.12, 0.10, 0.13, 0.09, 0.09, 0.06, 0.04, 0.02, 0.02]
    },

    "ELECTIVE_EXPLORATION": {
        "courses": ["CSE3800", "CSE3810", "CSE3550", "CSE3200", "CSE3250", "CSE3160", "CSE3802", "CSE3150"],
        "weights": [0.18, 0.15, 0.13, 0.13, 0.13, 0.12, 0.08, 0.08]
    },

    "SENIOR_DESIGN_MODE": {
        "courses": ["CSE4939W", "CSE4940", "CSE4950", "CSE4951", "CSE4997", "CSE4900"],
        "weights": [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
    },

    "CHALLENGE_HEAVY_SEMESTER": {
        "courses": ["CSE3500", "CSE4300", "CSE4705", "CSE3666", "CSE4302", "CSE4102", "CSE3502"],
        "weights": [0.22, 0.20, 0.18, 0.15, 0.12, 0.08, 0.05]
    },
}

# RESPONSES

GPA_RESPONSES = {
    "URGENT_INTERVENTION": (
        "Your GPA puts you at real academic risk. Don't wait — go to the SoC advising office this week. "
        "If you're still in the withdrawal window, dropping your hardest course is not failure, it's strategy. "
        "One bad semester doesn't define you but ignoring it will compound."
    ),
    "FOUNDATIONAL_HABIT_BUILDING": (
        "You don't have a knowledge problem yet, you have a system problem. "
        "Go to office hours once a week for your hardest course even when you think you don't need to, "
        "do practice problems the same day as lecture not the night before the exam, "
        "and find one other person in your major to study with. These habits compound fast."
    ),
    "TARGETED_COURSE_RECOVERY": (
        "One or two courses are dragging your GPA down — identify exactly which ones and treat them like emergencies. "
        "Get a tutor or form a study group specifically for those courses. "
        "Everything else is fine, don't let a fixable problem spread into a general confidence issue."
    ),
    "SUSTAIN_AND_OPTIMIZE": (
        "You're doing well — now make it sustainable. "
        "Spaced repetition for theory-heavy courses, starting projects the day they're assigned, "
        "and protecting sleep are what separate a 3.2 from a 3.6 over time. "
        "The students who maintain strong GPAs in SoC aren't studying more hours, they're studying smarter."
    ),
    "ACCELERATE_TO_OPPORTUNITIES": (
        "Your GPA opens doors — make sure you're walking through them. "
        "Apply for undergraduate research with a faculty member, look at the 5-Year BS/MS option if grad school interests you, "
        "and start building a project portfolio now. "
        "Strong grades without experiences to match them is a missed opportunity at this stage."
    ),
}

def course_response(label, courses):
    course_str = ", ".join(courses)
    responses = {
        "FOLLOW_CORE_SEQUENCE": (
            f"Don't jump ahead — the core sequence in SoC is ordered for a reason. "
            f"Lock in your foundations first. Based on your profile, focus on: {course_str}. "
            f"Students who skip prereqs or overload cores early almost always pay for it later."
        ),
        "REBALANCE_LOAD": (
            f"Your current course mix is too heavy and something is going to slip. "
            f"Swap one hard technical course for something more manageable this semester. "
            f"Consider: {course_str}. "
            f"A slightly lighter semester now protects your GPA more than pushing through a brutal one does."
        ),
        "CONCENTRATION_AI": (
            f"You're in the AI/ML track — these courses define your technical identity to employers and grad programs. "
            f"Give them your best energy, not what's left over after everything else. "
            f"Priority courses for you: {course_str}."
        ),
        "CONCENTRATION_SOFTWARE": (
            f"Software Engineering track courses are your priority right now. "
            f"These are what employers in SWE roles look for when they screen your transcript. "
            f"Focus on: {course_str}."
        ),
        "CONCENTRATION_SECURITY": (
            f"Security track — cryptography, network security, and the lab courses are directly career-relevant "
            f"and highly competitive to have on your resume. "
            f"Your priority courses: {course_str}."
        ),
        "CONCENTRATION_SYSTEMS": (
            f"Systems track — OS and architecture are your anchors. "
            f"Computer networks and embedded systems round out the track and are heavily weighted by systems-focused employers. "
            f"Your priority courses: {course_str}."
        ),
        "CONCENTRATION_MOBILE": (
            f"Mobile Computing track — your courses should be building toward deployable apps and real UI/UX work. "
            f"Employers in this space want to see projects, not just coursework, so treat every lab as a portfolio piece. "
            f"Your priority courses: {course_str}."
        ),
        "CONCENTRATION_NAVAL": (
            f"Naval Science and Technology concentration — your path sits at the intersection of systems, "
            f"networks, and mission-critical software. Reliability and security fundamentals matter here more than anywhere else. "
            f"Your priority courses: {course_str}."
        ),
        "CONCENTRATION_ALGORITHMS": (
            f"Algorithms and Theory track — this is the hardest concentration to do well in and the most respected. "
            f"Lean into proof-based thinking and don't skip the theory when it feels abstract; "
            f"it's exactly what separates candidates in top-tier technical interviews. "
            f"Your priority courses: {course_str}."
        ),
        "CONCENTRATION_BIOINFORMATICS": (
            f"Bioinformatics concentration — you're sitting at a rare intersection of CS and life sciences "
            f"that very few undergrads have. That cross-disciplinary depth is a genuine differentiator for research roles and grad school. "
            f"Your priority courses: {course_str}."
        ),
        "MAJOR_DATA_SCIENCE": (
            f"As a Data Science major your core is analytics, machine learning, and working with real datasets end to end. "
            f"Don't neglect the statistical foundations — employers and grad programs will probe those just as hard as your coding skills. "
            f"Your priority courses: {course_str}."
        ),
        "PREREQ_CATCH_UP": (
            f"You're missing prereqs that will block you — don't let this slide. "
            f"This covers both the CSE core sequence and the math chain (calc, linear algebra, diff eq, probability) "
            f"that nearly every upper-level CSE course requires. "
            f"Get these cleared first: {course_str}."
        ),
        "ENGINEERING_CORE": (
            f"As a CSE (BSE) student you have engineering-specific requirements that CS and DSE students don't take. "
            f"CSE2301, CSE3504, ECE2001, and the physics and upper math sequence are all BSE-only and have tight prereq chains — "
            f"missing one pushes everything back. Plan these early. "
            f"Your engineering core priorities: {course_str}."
        ),
        "ELECTIVE_EXPLORATION": (
            f"You have room to explore — use it deliberately. "
            f"Pick something genuinely outside your comfort zone; the students who stand out in interviews "
            f"often have one unexpected interest that makes them memorable. "
            f"Good options for you: {course_str}."
        ),
        "SENIOR_DESIGN_MODE": (
            f"Senior Design runs both semesters and takes more time than almost every student expects. "
            f"Build your schedule around it, not the other way around. "
            f"Your Senior Design project is also your best portfolio piece for job applications. "
            f"Relevant courses: {course_str}."
        ),
        "CHALLENGE_HEAVY_SEMESTER": (
            f"You have the profile to take on a challenging semester — lean into it. "
            f"Loading up on hard core courses now compounds your technical depth fast. "
            f"Target: {course_str}."
        ),
    }
    return responses.get(label, f"Recommended courses: {course_str}.")


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
    return [
        user["GPA"],
        user["YEAR"],
        major_map.get(user["MAJOR"], 0),
        concentration_map.get(user["CONCENTRATION"], 8),
    ]

def encode_course_input(user):
    return [
        user["YEAR"],
        major_map.get(user["MAJOR"], 0),
        concentration_map.get(user["CONCENTRATION"], 8),
        load_map.get(user["LOAD"], 1),
        course_type_map.get(user["COURSE_TYPE"], 0),
    ]


# TRAINING DATA
# ── JSON schema ──────────────────────────────────────────────────────────────
# gpa_data.json   → list of {"GPA": float, "YEAR": int, "MAJOR": str,
#                             "CONCENTRATION": str, "LABEL": str}
# course_data.json→ list of {"YEAR": int, "MAJOR": str, "CONCENTRATION": str,
#                             "LOAD": str, "COURSE_TYPE": str, "LABEL": str}
# ─────────────────────────────────────────────────────────────────────────────
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_gpa_data(path="gpa_data.json"):
    with open(path) as f:
        records = json.load(f)
    X = np.array([encode_gpa_input(r) for r in records])
    y = np.array([r["LABEL"] for r in records])
    return X, y

def load_course_data(path="course_data.json"):
    with open(path) as f:
        records = json.load(f)
    X = np.array([encode_course_input(r) for r in records])
    y = np.array([r["LABEL"] for r in records])
    return X, y

def train_and_validate(X, y, label="model", val_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=50, random_state=random_state)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc   = accuracy_score(y_val,   clf.predict(X_val))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Train size: {len(X_train)}  |  Val size: {len(X_val)}")
    print(f"  Train accuracy : {train_acc:.3f}")
    print(f"  Val   accuracy : {val_acc:.3f}")
    if train_acc - val_acc > 0.15:
        print("  ⚠  Gap > 15% — model may be overfitting. Add more data.")
    print(f"\n{classification_report(y_val, clf.predict(X_val), zero_division=0)}")
    print("="*60)

    return clf


# TRAIN MODELS
# Falls back to the small hardcoded arrays if JSON files are not found yet.
# Once you supply gpa_data.json and course_data.json, the JSON path is used.

_X_gpa_fallback = np.array([
    [3.8, 3, 1, 0], [3.6, 2, 0, 8], [2.4, 2, 0, 8], [3.9, 4, 1, 0],
    [2.8, 3, 1, 1], [3.2, 2, 1, 1], [3.7, 3, 1, 0], [2.6, 1, 0, 8],
    [1.8, 1, 0, 8], [1.5, 2, 1, 2],
])
_y_gpa_fallback = np.array([
    "ACCELERATE_TO_OPPORTUNITIES", "SUSTAIN_AND_OPTIMIZE",
    "FOUNDATIONAL_HABIT_BUILDING",  "ACCELERATE_TO_OPPORTUNITIES",
    "TARGETED_COURSE_RECOVERY",     "SUSTAIN_AND_OPTIMIZE",
    "ACCELERATE_TO_OPPORTUNITIES",  "FOUNDATIONAL_HABIT_BUILDING",
    "URGENT_INTERVENTION",          "URGENT_INTERVENTION",
])

_X_course_fallback = np.array([
    [1, 0, 8, 1, 0], [2, 1, 0, 1, 0], [3, 1, 0, 2, 0], [4, 1, 0, 2, 0],
    [2, 0, 8, 0, 1], [3, 1, 1, 1, 0], [3, 1, 2, 1, 0], [3, 1, 3, 1, 0],
    [2, 1, 4, 1, 1], [3, 1, 5, 1, 0], [3, 0, 6, 2, 0], [3, 2, 7, 1, 1],
    [2, 2, 8, 1, 0], [1, 0, 8, 1, 0], [1, 2, 8, 1, 0], [2, 1, 8, 1, 0],
])
_y_course_fallback = np.array([
    "FOLLOW_CORE_SEQUENCE",     "CONCENTRATION_AI",
    "CHALLENGE_HEAVY_SEMESTER", "SENIOR_DESIGN_MODE",
    "REBALANCE_LOAD",           "CONCENTRATION_SOFTWARE",
    "CONCENTRATION_SECURITY",   "CONCENTRATION_SYSTEMS",
    "CONCENTRATION_MOBILE",     "CONCENTRATION_NAVAL",
    "CONCENTRATION_ALGORITHMS", "CONCENTRATION_BIOINFORMATICS",
    "MAJOR_DATA_SCIENCE",       "PREREQ_CATCH_UP",
    "PREREQ_CATCH_UP",          "ENGINEERING_CORE",
])

if os.path.exists("gpa_data.json"):
    print("[INFO] Loading GPA data from gpa_data.json")
    X_gpa, y_gpa = load_gpa_data("gpa_data.json")
    gpa_model = train_and_validate(X_gpa, y_gpa, label="GPA MODEL")
else:
    print("[INFO] gpa_data.json not found — using fallback data (no validation split)")
    gpa_model = RandomForestClassifier(n_estimators=50, random_state=42)
    gpa_model.fit(_X_gpa_fallback, _y_gpa_fallback)

if os.path.exists("course_data.json"):
    print("[INFO] Loading course data from course_data.json")
    X_course, y_course = load_course_data("course_data.json")
    course_model = train_and_validate(X_course, y_course, label="COURSE MODEL")
else:
    print("[INFO] course_data.json not found — using fallback data (no validation split)")
    course_model = RandomForestClassifier(n_estimators=50, random_state=42)
    course_model.fit(_X_course_fallback, _y_course_fallback)



# PREDICT + SAMPLE
def predict_gpa_label(user):
    probs = gpa_model.predict_proba([encode_gpa_input(user)])[0]
    return dict(zip(gpa_model.classes_, probs))

def predict_course_label(user):
    probs = course_model.predict_proba([encode_course_input(user)])[0]
    return dict(zip(course_model.classes_, probs))

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
    weights = weights / weights.sum()
    return list(np.random.choice(courses, size=k, replace=False, p=weights))


# PIPELINES
def run_gpa_help(user):
    label_probs = predict_gpa_label(user)
    top_label   = max(label_probs, key=label_probs.get)
    return GPA_RESPONSES[top_label]

def run_course_selection(user):
    label_probs  = predict_course_label(user)
    top_label    = max(label_probs, key=label_probs.get)
    course_dist  = build_course_distribution(label_probs)
    courses      = sample_from_distribution(course_dist, k=3)
    return course_response(top_label, courses)


# ENTRY POINT

def recommend(inputs):
    topic = inputs[0].lower()
    if topic == "gpa_help":
        _, gpa, major, concentration, year = inputs
        return run_gpa_help({"GPA": gpa, "MAJOR": major, "CONCENTRATION": concentration, "YEAR": year})
    elif topic == "course_selection":
        _, major, concentration, load, course_type, year = inputs
        return run_course_selection({"MAJOR": major, "CONCENTRATION": concentration, "LOAD": load, "COURSE_TYPE": course_type, "YEAR": year})
    else:
        raise ValueError(f"Unknown topic '{inputs[0]}'. Use 'gpa_help' or 'course_selection'.")


# TEST RUN
if __name__ == "__main__":
    tests = [
        # GPA tests
        ["gpa_help",         1.8,  "CSE", "None",                           2],
        ["gpa_help",         2.5,  "CS",  "Software Design and Development", 3],
        ["gpa_help",         3.5,  "DSE", "None",                           3],
        ["gpa_help",         3.9,  "CSE", "Artificial Intelligence",        4],
        # Concentration tests
        ["course_selection", "CSE", "Artificial Intelligence",              "Hard",   "Core",     3],
        ["course_selection", "CS",  "Cybersecurity",                        "Light",  "Elective", 2],
        ["course_selection", "CSE", "Systems and Networks",                 "Medium", "Core",     4],
        ["course_selection", "DSE", "None",                                 "Medium", "Core",     3],
        ["course_selection", "CS",  "Algorithms and Theory",                "Hard",   "Core",     3],
        ["course_selection", "CSE", "Bioinformatics",                       "Medium", "Elective", 2],
        ["course_selection", "CSE", "Software Design for Mobile Computing", "Medium", "Elective", 3],
        ["course_selection", "CSE", "Naval Science and Technology",         "Medium", "Core",     3],
        # New label tests — PREREQ, MATH, ENGINEERING_CORE
        ["course_selection", "CS",  "None",                                 "Medium", "Core",     1],
        ["course_selection", "DSE", "None",                                 "Light",  "Core",     1],
        ["course_selection", "CSE", "None",                                 "Medium", "Core",     2],
    ]
    for t in tests:
        print(f"\nInput: {t}")
        print(recommend(t))
        print("-" * 60)