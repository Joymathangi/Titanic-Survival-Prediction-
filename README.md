
# Titanic Survival Prediction

## Objectives
Predict Titanic passenger survival using `tested.csv` with features like age, gender, and class.

## Steps to Run
1. Clone: `git clone https://github.com/Joymathangi/Titanic-Survival-Prediction-`
2. Install: `pip install pandas numpy scikit-learn matplotlib`
3. Place `tested.csv` in the folder.
4. Run: `python titanic_prediction.py`

## Implementation
- **Preprocessing**: Filled missing `Age`/`Fare`, dropped `Cabin`, encoded `Sex`/`Embarked`, normalized data.
- **Model**: Random Forest Classifier.
- **Results**: Accuracy: 1.00, Precision: 1.00. See `feature_importance.png`.
