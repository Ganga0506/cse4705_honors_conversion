from ..nn_classifier import classify

unseen = [
    # -----------------------
    # course_recommendation (VERY diverse)
    # -----------------------
    ("i dont think i can handle another heavy semester", "course_recommendation"),
    ("trying to avoid burning out next term", "course_recommendation"),
    ("what should i take if i want things to be chill", "course_recommendation"),
    ("i need a better balance in my schedule", "course_recommendation"),
    ("my last semester was rough i need something lighter", "course_recommendation"),
    ("how do people usually plan their classes", "course_recommendation"),
    ("i want courses that wont stress me out too much", "course_recommendation"),
    ("what fits well if im already taking hard classes", "course_recommendation"),
    ("any suggestions for not overloading myself", "course_recommendation"),
    ("im confused about what to register for", "course_recommendation"),
    ("i dont wanna regret my classes again", "course_recommendation"),
    ("what works for someone with a busy life", "course_recommendation"),
    ("how do i pick classes without messing up my gpa", "course_recommendation"),
    ("what would you take in my position", "course_recommendation"),
    ("i just need something manageable honestly", "course_recommendation"),

    # -----------------------
    # study_habits (VERY diverse)
    # -----------------------
    ("i feel lost in all my classes", "study_habits"),
    ("nothing is clicking for me", "study_habits"),
    ("i sit down to study and get nowhere", "study_habits"),
    ("i keep falling behind no matter what", "study_habits"),
    ("i dont know how to approach studying anymore", "study_habits"),
    ("im putting in hours but its not working", "study_habits"),
    ("i cant stay consistent with anything", "study_habits"),
    ("every exam feels like a disaster", "study_habits"),
    ("i blank out during tests", "study_habits"),
    ("i dont even know where to start studying", "study_habits"),
    ("i keep getting distracted every few minutes", "study_habits"),
    ("im trying but my results dont show it", "study_habits"),
    ("i feel overwhelmed trying to study everything", "study_habits"),
    ("i dont retain anything after i read it", "study_habits"),
    ("i procrastinate until its too late every time", "study_habits"),

    # -----------------------
    # unknown (VERY diverse)
    # -----------------------
    ("whats a good place to eat nearby", "unknown"),
    ("recommend me a netflix show", "unknown"),
    ("how do i fix my sleep schedule fast", "unknown"),
    ("whats happening in the world right now", "unknown"),
    ("tell me something random", "unknown"),
    ("what should i do for fun today", "unknown"),
    ("im bored out of my mind", "unknown"),
    ("how do i cook eggs properly", "unknown"),
    ("what is a good workout plan", "unknown"),
    ("how do i meet new people", "unknown"),
    ("do you know any fun facts", "unknown"),
    ("what music should i listen to", "unknown"),
    ("whats trending right now", "unknown"),
    ("how do i get better at gaming", "unknown"),
    ("give me something interesting", "unknown"),

    # -----------------------
    # EXTRA HARD (ambiguous / tricky)
    # -----------------------
    ("my semester is killing me", "study_habits"),
    ("i hate how my classes are going", "study_habits"),
    ("this schedule is too much for me", "course_recommendation"),
    ("im overwhelmed with everything right now", "study_habits"),
    ("i regret what i signed up for", "course_recommendation"),
    ("i cant keep up anymore", "study_habits"),
    ("this is way harder than i expected", "study_habits"),
    ("i dont think i chose the right classes", "course_recommendation"),
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