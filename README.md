# House Price Prediction

Predicts median house values using the California Housing dataset.  
Compares Linear Regression (baseline) vs Random Forest.

## Results
| Model | R² | RMSE |
|---|---|---|
| Linear Regression | 0.60 | 0.73 |
| Random Forest | 0.81 | 0.51 |

## Key Insight
Median income (MedInc) is the strongest predictor of house price (correlation: 0.69).

## Tech Stack
Python · Scikit-learn · Pandas · Matplotlib · Seaborn

## Run it
pip install -r requirements.txt
python model.py
