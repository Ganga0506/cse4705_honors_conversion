from ..nn_classifier import classify

unseen = [
    # should be course_recommendation
    ("im not sure what to take next semester", "course_recommendation"),
    ("what courses wont destroy my gpa",       "course_recommendation"),
    ("help me figure out my schedule",         "course_recommendation"),
    ("what should a busy junior take",         "course_recommendation"),

    # should be study_habits
    ("i bombed my midterm",                    "study_habits"),
    ("i cant get myself to open my textbook",  "study_habits"),
    ("my professor moves too fast",            "study_habits"),
    ("i study but nothing sticks",             "study_habits"),

    # should be unknown
    ("whats the weather today",                "unknown"),
    ("tell me a joke",                         "unknown"),
    ("who is the president",                   "unknown"),
]

correct = 0
for sentence, expected in unseen:
    result = classify(sentence)
    predicted = result["intent"]
    conf = result["confidence"]
    status = "PASS" if predicted == expected else "FAIL"
    if status == "PASS":
        correct += 1
    print(f"{status} | expected: {expected:25s} got: {predicted:25s} conf: {conf} | '{sentence}'")

print(f"\n{correct}/{len(unseen)} correct on unseen sentences")