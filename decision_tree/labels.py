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
    "PREREQ_CATCH_UP": {
        "courses": [
            "CSE1729", "CSE2050", "CSE2500", "CSE3500",
            "MATH1131Q", "MATH1132Q", "MATH2110Q", "MATH2210Q", "MATH2410Q",
            "MATH3160", "STAT3025Q", "STAT3345Q",
        ],
        "weights": [0.15, 0.13, 0.10, 0.08, 0.12, 0.10, 0.09, 0.09, 0.07, 0.04, 0.02, 0.01]
    },
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