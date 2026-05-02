import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from decision_tree import recommend

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
