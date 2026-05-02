# Decision Tree Chatbot — Academic Advising Component

> **Course project:** Comparative analysis of a rule-based / tree-based advising chatbot vs. a neural network classifier.
> This component is the **decision tree side** of that comparison. It is designed to demonstrate both the engineering skills involved in building a probabilistic tree-based pipeline *and* the fundamental limitations that make this approach fail relative to a neural network on real advising data.

---

## What This Component Does

This is a Random Forest-based chatbot pipeline that takes a student profile as input and returns one of two types of advising output:

- **GPA Help** — classifies the student's academic situation into one of 5 intervention labels and returns targeted advising text
- **Course Recommendation** — classifies the student's profile into one of 16 course strategy labels and returns a weighted, probabilistic course suggestion

The entry point is `decision_tree.py → recommend(inputs)`, which routes to either pipeline based on the topic argument.

---

## File Structure

```
decision_tree/
├── decision_tree.py       # Core encoding, model loading, prediction, recommend()
├── training.py            # Model training, validation, saves .pkl files
├── labels.py              # Label definitions, course maps, response text
├── testing.py             # Manual smoke tests
├── eval/
│   └── eval.py            # Full ML evaluation (CV metrics, plots)
│   └── audit_data.py      # Diagnoses whether training data is synthetic
└── d_data/
    ├── gpa_data.json       # GPA training data
    └── course_data.json    # Course training data
```

---

## Skills Demonstrated

### 1. Probabilistic Output Layer (not just argmax)
Rather than taking the single top predicted class, the course recommendation pipeline uses the full `predict_proba` distribution across all 16 labels to build a weighted course pool:

```python
def build_course_distribution(label_probs):
    combined = {}
    for label, prob in label_probs.items():
        courses = course_map[label]["courses"]
        weights = course_map[label]["weights"]
        for c, w in zip(courses, weights):
            combined[c] = combined.get(c, 0) + prob * w
    return combined
```

When the model is uncertain (top label confidence ≤ 0.6), it samples from this blended distribution rather than committing to a single label's course list. This makes the output more robust near decision boundaries — a deliberate design choice to partially compensate for the model's overconfidence on synthetic data.

### 2. Feature Encoding Pipeline
Raw student profile fields (major, concentration, load, course type) are encoded into a fixed numeric vector before hitting the model. Two separate encoders handle the GPA and course pipelines since they use different feature sets:

```python
GPA_FEATURES    = ["GPA", "YEAR", "MAJOR", "CONCENTRATION"]
COURSE_FEATURES = ["YEAR", "MAJOR", "CONCENTRATION", "LOAD", "COURSE_TYPE"]
```

### 3. Full ML Evaluation Suite (`eval.py`)
Rather than just reporting train/val accuracy, the eval script produces:
- 5-fold stratified cross-validation (avoids train/test leakage)
- Per-class classification report (precision, recall, F1)
- Confusion matrix
- ROC curves (one-vs-rest, with macro average)
- Precision-Recall curves with average precision
- Calibration / reliability diagrams
- Feature importances

### 4. Data Auditing (`audit_data.py`)
A diagnostic script that checks whether training data is synthetic by looking for non-overlapping GPA ranges, perfectly balanced class counts, and deterministic feature-to-label mappings. This was used to understand and document the failure mode described below.

### 5. Weighted Course Sampling
Within each label, courses are not selected uniformly — each has a priority weight defined in `labels.py`. The sampling respects these weights, so higher-priority courses appear more often in recommendations.

---

## Why This Approach Fails — The Point of the Experiment

### The Metrics Look Perfect (That's the Problem)

| Model  | CV Accuracy | Mean AUC | Mean AP |
|--------|-------------|----------|---------|
| GPA    | 1.0000      | 1.00     | 1.00    |
| Course | 1.0000      | 1.00     | 1.00    |

Every metric is perfect. Every confusion matrix is a clean diagonal. This is not a success — **it is the failure mode made visible.**

### Root Cause 1: Synthetic, Rule-Based Training Data
The training data was programmatically generated with deterministic label assignment:
- GPA labels are assigned by hard GPA thresholds with no overlap (e.g. GPA < 2.0 → `URGENT_INTERVENTION`, always)
- Course labels are assigned by concentration alone — every student with `Artificial Intelligence` concentration gets `CONCENTRATION_AI`, regardless of year, load, or course type

The Random Forest doesn't learn anything meaningful. It memorizes the same rules that generated the data. A lookup table would perform identically.

### Root Cause 2: Feature Importance Collapse
The feature importance plots confirm this directly:

- **GPA model**: GPA alone accounts for ~99% of importance. YEAR, MAJOR, and CONCENTRATION contribute essentially nothing.
- **Course model**: CONCENTRATION accounts for ~45%, YEAR ~30%. COURSE_TYPE registers near zero — it has no predictive signal in the data.

### Root Cause 3: Calibration Breakdown
The calibration plots show the model outputs probabilities of either 0 or 1 with no uncertainty. On the course model, every class collapses to just two points — (0, 0) and (1, 1). A well-calibrated model should distribute predicted probabilities smoothly along the diagonal. This model is not probabilistic in any meaningful sense; it is a hard classifier pretending to output probabilities.

### Why a Neural Network Does Better on This Task
A properly trained neural network (see companion component) can:
- Learn non-linear feature interactions (e.g. how GPA + year + load together predict risk)
- Generalize to student profiles that fall between label boundaries
- Produce calibrated uncertainty estimates that reflect genuine ambiguity
- Be trained on real advising data where labels are not deterministic

The decision tree approach is fundamentally limited here because the task requires understanding combinations of features, not just finding a single dominant split variable. When GPA alone determines the label, a tree will find that split immediately and ignore everything else — which is exactly what the feature importance plot shows.

---

## How to Run

### Train models
```bash
python training.py
```
Trains both models, prints validation metrics, saves `gpa_model.pkl` and `course_model.pkl`.

### Test the recommend() pipeline
```bash
python testing.py
```

### Run full evaluation
```bash
cd eval
python eval.py
# or with custom data directory:
python eval.py --data-dir path/to/d_data
```
Outputs plots to `eval/eval_output/`.

### Audit training data
```bash
python audit_data.py
```

---

## Conclusion

This component succeeds as an engineering exercise — it implements a full ML pipeline with probabilistic output, weighted sampling, cross-validated evaluation, and diagnostic tooling. The probabilistic course distribution layer in particular is a meaningful improvement over naive argmax classification.

It fails as an advising system for the reasons documented above: synthetic deterministic training data makes perfect metrics meaningless, feature importance collapses to a single variable, and the model learns rules rather than patterns. That failure is the intended result of this experiment, and the comparison with the neural network component demonstrates why representation learning is better suited to this kind of advising task.
