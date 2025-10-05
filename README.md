# Sifting ExoS – Exoplanet Classification Using NASA Kepler Data

### Team Information
**Team Name:** KrishnVivar 
**Team Member:** Aryan Maurya
**Challenge Selected:** A World Away: Hunting for Exoplanets with AI

---

## Project Overview

Sifting ExoS is a machine learning application that identifies whether a detected celestial body is a **confirmed exoplanet**, **candidate**, or **false positive**.  
It utilises **NASA’s Kepler Object of Interest (KOI)** dataset to analyse parameters such as flux ratio, signal-to-noise ratio, planet radius, impact parameter, and various false positive flags.

The model was trained using a **Random Forest Classifier**, achieving an accuracy of approximately **80%** after balancing and feature optimization.  
The project demonstrates how open NASA space data can be transformed into a user-friendly AI-powered application for exoplanet research.

---

## Features

- Predicts the **disposition** of a detected celestial object.  
- Displays the **probability distribution** for each class (Confirmed, Candidate, False Positive).  
- Provides a **data-driven reasoning** behind predictions.  
- Simple **Streamlit web interface** for live testing.  

---

## Files and Structure

| File/Folder | Description |
|--------------|-------------|
| [app.py](./app.py) | Streamlit web app for exoplanet prediction |
| [main.py](./main.py) | Model training and feature selection script |
| [siftingexosaccurate_model.pkl](./siftingexosaccurate_model.pkl) | Trained Random Forest model |
| [KOI_CSV.csv](./KOI_CSV.csv) | NASA Kepler Object of Interest dataset used for training |
| This dataset was downloaded from NASA's Website: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi
| [Images](./Images) | Contains outputs and Web Interface of our model |
| [README.md](./README.md) | Documentation file (this file) |

---

## User Experience

1. Run the app with:
   ```bash
   streamlit run app.py
   
2. Enter the 9 required input parameters.

3. The app displays the prediction (Confirmed, Candidate, or False Positive) along with confidence percentages and reasoning.
