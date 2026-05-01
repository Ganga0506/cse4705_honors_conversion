# Academic Advisor Chatbot

Academic Advisor Chatbot is a hybrid ML project that helps students with two tasks: course recommendations and GPA improvement guidance.  
It combines an intent classifier with slot extraction to understand user requests, then routes structured inputs into a decision system built with Random Forest models.  
A lightweight Flask frontend provides a simple chat interface for interacting with the pipeline in the browser.  
The project is designed as a practical academic advising prototype with modular components.

## Architecture Overview

- **Neural Network (Intent Classification)**: Classifies user messages into supported intents (for example, course recommendation vs GPA help).
- **Slot Extraction**: Uses regex and keyword matching to pull fields like GPA, major, year, concentration, load preference, and course type.
- **Decision Tree / Random Forest System**: Generates GPA advice and course recommendations from structured slot values.
- **Flask Frontend**: Exposes a basic web chat UI and connects user input to the end-to-end ML pipeline.

## How It Works

`User input → Intent classification → Slot extraction → Follow-up questions (if needed) → Decision system → Response`

## Setup

1. **Clone the repository** and open it in your terminal.
2. **Create and activate a virtual environment** (recommended):
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
   - - Windows: 
      - `.venv\Scripts\activate`
3. **Install dependencies**:
   - `pip install -r requirements.txt`
4. **Run the Flask app**:
   - `python main.py`
5. Open your browser at:
   - `http://127.0.0.1:5000`


**Curious to see our training and evualtions? Train decision models and nerual networks**:
   - `python decision_tree/training.py`
   - `python -m neural_network.train`
   - `python -m neural_network.nn_classifier` 
   - `python -m neural_network.eval.eval_confidence`
   - `python -m neural_network.eval.eval_e2e`
   - `python -m neural_network.eval.eval_nn`
   - `python -m neural_network.eval.eval_slots`
   - `python -m neural_network.eval.eval_unseen`   

## Project Structure

```text
cse4705_honors_conversion/
├── main.py
├── requirements.txt
├── README.md
├── chatbot_model.pth
├── neural_network/
│   ├── model.py
│   ├── preprocess.py
│   ├── slot_extractor.py
│   ├── nn_classifier.py
│   ├── train.py
│   ├── nn_data/
│   │   └── intents.json
│   └── eval/
├── decision_tree/
│   ├── decision_tree.py
│   ├── training.py
│   ├── labels.py
│   ├── d_data/
│   │   ├── gpa_data.json
│   │   └── course_data.json
│   ├── gpa_model.pkl
│   └── course_model.pkl
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Dependencies

All required third-party packages are listed in `requirements.txt`.
