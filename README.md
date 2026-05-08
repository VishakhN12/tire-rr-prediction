# Tire Rolling Resistance Prediction using MATLAB

## Project Overview
This project predicts tire rolling resistance (RR_80KMPH) using machine learning models based on tire geometry and operational parameters.

## Dataset
- Tire dimensions (OD, SW, TW)
- Inflation parameters (PSI)
- Load capacity
- Contact patch indicators
- Rolling resistance at 80 km/h (target)

## Methodology
- Data preprocessing using MATLAB
- Feature engineering (deformation-based features)
- Linear regression model
- Ensemble Bagging model
- Train-test split (80/20)
- Performance evaluation (RMSE, R²)

## Models Used
- Linear Regression (fitlm)
- Bagging Ensemble Regression (fitrensemble)

## Key Insight
Geometry-only features provide limited predictive accuracy for rolling resistance, indicating the importance of material and thermal properties in tire performance modeling.

## Results
- Linear R² ≈ 0.20
- Ensemble R² ≈ 0.18

## Tools Used
- MATLAB 2023b
- Statistics and Machine Learning Toolbox

## Author
Vish
