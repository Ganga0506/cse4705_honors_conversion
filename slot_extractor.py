import re

# ---- Lookup maps ----
MAJORS = {
    "cs": "CS",
    "computer science": "CS",
    "cse": "CSE",
    "computer science and engineering": "CSE",
    "computer engineering": "CSE",
    "dse": "DSE",
    "ds": "DSE",
    "data science": "DSE"
}

CONCENTRATIONS = {
    # Algorithms and Theory
    "algorithms": "Algorithms and Theory",
    "theory": "Algorithms and Theory",
    "algorithms and theory": "Algorithms and Theory",

    # Artificial Intelligence
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "machine learning": "Artificial Intelligence",
    "ml": "Artificial Intelligence",

    # Bioinformatics
    "bioinformatics": "Bioinformatics",
    "bio": "Bioinformatics",
    "computational biology": "Bioinformatics",

    # Naval Science and Technology
    "naval": "Naval Science and Technology",
    "naval science": "Naval Science and Technology",
    "navy": "Naval Science and Technology",
    "naval technology": "Naval Science and Technology",

    # Cybersecurity
    "cybersecurity": "Cybersecurity",
    "cyber": "Cybersecurity",
    "security": "Cybersecurity",
    "infosec": "Cybersecurity",

    # Software Design and Development
    "software": "Software Design and Development",
    "software design": "Software Design and Development",
    "software development": "Software Design and Development",
    "software engineering": "Software Design and Development",

    # Software Design for Mobile Computing
    "mobile": "Software Design for Mobile Computing",
    "mobile computing": "Software Design for Mobile Computing",
    "mobile development": "Software Design for Mobile Computing",
    "mobile apps": "Software Design for Mobile Computing",
    "android": "Software Design for Mobile Computing",
    "ios": "Software Design for Mobile Computing",

    # Systems and Networks
    "systems": "Systems and Networks",
    "networks": "Systems and Networks",
    "systems and networks": "Systems and Networks",
    "networking": "Systems and Networks",
    "distributed systems": "Systems and Networks"
}

LOAD = {
    "light": "Light", "easy": "Light", "lighter": "Light",
    "medium": "Medium", "moderate": "Medium",
    "hard": "Hard", "harder": "Hard", "challenging": "Hard"
}

COURSE_TYPE = {
    "core": "Core", "required": "Core", "requirement": "Core",
    "elective": "Elective", "electives": "Elective", "optional": "Elective"
}

YEAR_WORDS = {
    "freshman": 1, "first year": 1, "1st year": 1,
    "sophomore": 2, "second year": 2, "2nd year": 2,
    "junior": 3, "third year": 3, "3rd year": 3,
    "senior": 4, "fourth year": 4, "4th year": 4
}

# ---- Extractor functions ----
def extract_gpa(text):
    match = re.search(r'\b([0-4]\.\d{1,2})\b', text)
    if match:
        gpa = float(match.group(1))
        if 0.0 <= gpa <= 4.0:
            return gpa
    return None

def extract_year(text):
    text_lower = text.lower()
    for phrase, yr in YEAR_WORDS.items():
        if phrase in text_lower:
            return yr
    match = re.search(r'\byear\s*([1-4])\b|\b([1-4])(st|nd|rd|th)\s*year\b', text_lower)
    if match:
        return int(match.group(1) or match.group(2))
    return None

def extract_major(text):
    text_lower = text.lower()
    for key, val in MAJORS.items():
        if key in text_lower:
            return val
    return None

def extract_concentration(text):
    text_lower = text.lower()
    for key, val in CONCENTRATIONS.items():
        if key in text_lower:
            return val
    return None

def extract_load(text):
    text_lower = text.lower()
    for key, val in LOAD.items():
        if key in text_lower:
            return val
    return None

def extract_course_type(text):
    text_lower = text.lower()
    for key, val in COURSE_TYPE.items():
        if key in text_lower:
            return val
    return None

def extract_slots(sentence, intent):
    slots = {}

    if intent == "study_habits":
        slots["gpa"]           = extract_gpa(sentence)
        slots["major"]         = extract_major(sentence)
        slots["concentration"] = extract_concentration(sentence)
        slots["year"]          = extract_year(sentence)

    elif intent == "course_recommendation":
        slots["major"]           = extract_major(sentence)
        slots["concentration"]   = extract_concentration(sentence)
        slots["year"]            = extract_year(sentence)
        slots["load_preference"] = extract_load(sentence)
        slots["course_type"]     = extract_course_type(sentence)

    return slots


if __name__ == "__main__":
    tests = [
        ("I have a 2.3 GPA and I am a sophomore CS student", "study_habits"),
        ("I am a junior in CSE with AI concentration, recommend hard electives", "course_selection"),
        ("my GPA is 3.1 and I do data science", "study_habits"),
        ("I want light core courses for my first year", "course_selection"),
    ]
    for sentence, intent in tests:
        slots = extract_slots(sentence, intent)
        print(f"\nInput:  '{sentence}'")
        print(f"Intent: {intent}")
        print(f"Slots:  {slots}")