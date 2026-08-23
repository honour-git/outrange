# Golf Ball Trajectory Prediction from Netted Range Data

> **Inrange® Student Competition Submission**  
> **Author:** Hlompho Honour Monyela
> **Affiliation:** Stellenbosch University (BSc Computer Science, 3rd Year)  

---

## Project Overview
This repository contains a hybrid machine learning and physics-informed model to predict full golf ball trajectories (spin rate, apex position/time, landing position/time) using partial measured data from a netted driving range (launch conditions + 4 checkpoint crossings up to 60 meters).

The solution combines:
1. **Physics-Based Differential Equations (`scipy`):** Fitting drag ($C_d$) and lift/Magnus effect ($C_l$) aerodynamic models to initial checkpoint velocity vectors.
2. **Gradient-Boosted Decision Trees (LightGBM / CatBoost):** Predicting non-linear output targets directly from engineered kinematic features.
3. **Ensemble Blending:** Combining physics-informed projections with empirical model predictions to minimize cross-validation error.

---

## Project Structure

```text
├── data/                  # raw data files (added to .gitignore)
├── notebooks/             # exploratory data analysis (EDA) and visualizations
├── src/
│   ├── physics.py         # 3D projectile motion ODE solver and aerodynamic models
│   ├── features.py        # kinematic feature generation (velocity deltas, curvature)
│   └── models.py          # model training, hyperparameter tuning & evaluation
├── outputs/               # generated submission.csv files
├── train.py               # end-to-end execution script
├── requirements.txt       # project dependencies
└── README.md              # documentation and overview