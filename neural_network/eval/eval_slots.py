from neural_network.slot_extractor import extract_slots

tests = [
    ("I am a junior CSE student with AI concentration", "course_recommendation",
     {"year": 3, "major": "CSE", "concentration": "Artificial Intelligence"}),
    ("my GPA is 3.2 and I am a sophomore", "study_habits",
     {"gpa": 3.2, "year": 2}),
    ("I want hard electives for data science", "course_recommendation",
     {"load_preference": "Hard", "course_type": "Elective", "concentration": "None"}),
    ("I am a DSE major", "course_recommendation",
     {"major": "DSE"}),
    ("no concentration", "course_recommendation",
     {"concentration": "None"}),
    ("my GPA is 3", "study_habits",
     {"gpa": 3.0}),
]

passed = 0
total  = 0
for sentence, intent, expected in tests:
    result = extract_slots(sentence, intent)
    for key, val in expected.items():
        total += 1
        if result.get(key) == val:
            print(f"  PASS — {key}: {val}")
            passed += 1
        else:
            print(f"  FAIL — {key}: expected {val}, got {result.get(key)}")

print(f"\n{passed}/{total} slots correct")