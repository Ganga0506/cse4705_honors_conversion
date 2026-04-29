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

    "CONCENTRATION_DATA": {
        "courses": ["CSE4502", "CSE4701", "CSE2600", "CSE4820", "CSE3800", "CSE3810", "CSE3802"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "CONCENTRATION_SYSTEMS": {
        "courses": ["CSE4300", "CSE4302", "CSE3300", "CSE3100", "CSE3666", "CSE4709", "CSE3504"],
        "weights": [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
    },

    "PREREQ_CATCH_UP": {
        "courses": ["CSE1729", "CSE2050", "CSE2301", "CSE2500", "CSE3500"],
        "weights": [0.30, 0.25, 0.20, 0.15, 0.10]
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
        "CONCENTRATION_DATA": (
            f"Data Science track — center your semester on analytics, databases, and ML. "
            f"Bioinformatics courses are a strong differentiator for this track in interviews. "
            f"Focus on: {course_str}."
        ),
        "CONCENTRATION_SYSTEMS": (
            f"Systems track — OS and architecture are your anchors. "
            f"Computer networks and embedded systems round out the track and are heavily weighted by systems-focused employers. "
            f"Your priority courses: {course_str}."
        ),
        "PREREQ_CATCH_UP": (
            f"You're behind on a prereq that's going to block you — fix this now before it cascades. "
            f"Missing one prereq can push your entire concentration sequence back a full semester. "
            f"Get these done first: {course_str}."
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
concentration_map = {"AI": 0, "Software": 1, "Security": 2, "DataScience": 3, "Systems": 4, "None": 5}
load_map          = {"Light": 0, "Medium": 1, "Hard": 2}

def encode_gpa_input(user):
    return [
        user["GPA"],
        user["YEAR"],
        major_map.get(user["MAJOR"], 0),
        concentration_map.get(user["CONCENTRATION"], 5),
    ]

def encode_course_input(user):
    return [
        user["YEAR"],
        major_map.get(user["MAJOR"], 0),
        concentration_map.get(user["CONCENTRATION"], 5),
        load_map.get(user["LOAD"], 1),
    ]


# TRAINING DATA
X_gpa = np.array([
    [3.8, 3, 1, 0], [3.6, 2, 0, 5], [2.4, 2, 0, 5], [3.9, 4, 1, 0],
    [2.8, 3, 1, 1], [3.2, 2, 1, 1], [3.7, 3, 1, 0], [2.6, 1, 0, 5],
    [1.8, 1, 0, 5], [1.5, 2, 1, 2],
])
y_gpa = np.array([
    "ACCELERATE_TO_OPPORTUNITIES", "SUSTAIN_AND_OPTIMIZE",
    "FOUNDATIONAL_HABIT_BUILDING", "ACCELERATE_TO_OPPORTUNITIES",
    "TARGETED_COURSE_RECOVERY",    "SUSTAIN_AND_OPTIMIZE",
    "ACCELERATE_TO_OPPORTUNITIES", "FOUNDATIONAL_HABIT_BUILDING",
    "URGENT_INTERVENTION",         "URGENT_INTERVENTION",
])

X_course = np.array([
    [1, 0, 5, 1], [2, 1, 0, 1], [3, 1, 0, 2], [4, 1, 0, 2],
    [2, 0, 5, 0], [3, 1, 1, 1], [3, 1, 2, 1], [3, 2, 3, 1],
    [3, 1, 4, 1], [2, 0, 5, 1],
])
y_course = np.array([
    "FOLLOW_CORE_SEQUENCE",    "CONCENTRATION_AI",
    "CHALLENGE_HEAVY_SEMESTER","SENIOR_DESIGN_MODE",
    "REBALANCE_LOAD",          "CONCENTRATION_SOFTWARE",
    "CONCENTRATION_SECURITY",  "CONCENTRATION_DATA",
    "CONCENTRATION_SYSTEMS",   "ELECTIVE_EXPLORATION",
])


# TRAIN MODELS
gpa_model = RandomForestClassifier(n_estimators=50, random_state=42)
gpa_model.fit(X_gpa, y_gpa)

course_model = RandomForestClassifier(n_estimators=50, random_state=42)
course_model.fit(X_course, y_course)


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
        _, major, concentration, load, year = inputs
        return run_course_selection({"MAJOR": major, "CONCENTRATION": concentration, "LOAD": load, "YEAR": year})
    else:
        raise ValueError(f"Unknown topic '{inputs[0]}'. Use 'gpa_help' or 'course_selection'.")


# TEST RUN
if __name__ == "__main__":
    tests = [
        ["gpa_help",          1.8,  "CSE", "AI",          2],
        ["gpa_help",          2.5,  "CS",  "Software",    3],
        ["gpa_help",          3.5,  "DSE", "DataScience", 3],
        ["gpa_help",          3.9,  "CSE", "AI",          4],
        ["course_selection",  "CSE", "AI",          "Hard",   3],
        ["course_selection",  "CS",  "Security",    "Light",  2],
        ["course_selection",  "CSE", "Systems",     "Medium", 4],
        ["course_selection",  "CS",  "DataScience", "Medium", 3],
    ]
    for t in tests:
        print(f"\nInput: {t}")
        print(recommend(t))
        print("-" * 60)