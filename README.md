# Sifting ExoS – Exoplanet Classification Using NASA Kepler Data

### Team Information
**Team Name:** KrishnVivar | 
**Team Member:** Aryan Maurya | 
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
| https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi | This dataset was downloaded from NASA's Website |
| [Images](./Images) | Contains outputs and Web Interface of our model |
| [README.md](./README.md) | Documentation file (this file) |

---

## User Experience

1. Run the app with:
   ```bash
   streamlit run app.py
   
2. Enter the 9 required input parameters.

3. The app displays the prediction (Confirmed, Candidate, or False Positive) along with confidence percentages and reasoning.

### NASA Open Data and Technology
This project is built using:
- NASA Exoplanet Archive (Kepler KOI Dataset)
- Python for machine learning and data preprocessing
- Streamlit for visualisation and deployment

By leveraging publicly available NASA data, we bridge scientific research and accessible technology.

### Model Details

*Algorithm: Random Forest Classifier*

Accuracy: *~80%*

Features Used:

- fpflag_ec
- model_snr
- prad
- impact
- fpflag_ss
- fpflag_co
- koi_duration
- koi_depth
- koi_period

### Hosting and Access

The app and project files are hosted on GitHub for public access.

Streamlit:
Local URL: http://localhost:8501

### Future Work

- Integrate with raw light curve data from TESS/Kepler for end-to-end automation.
- Improve accuracy using ensemble and deep learning models.
- Add explainable AI (XAI) to show reasoning behind predictions.
- Enable real-time updates from new NASA datasets.
- Build interactive dashboards for visualising light curves and results.
- Expand to other detection methods like radial velocity or microlensing.
- Support user uploads and retraining for continuous learning.
- Include an educational mode for students and astronomy enthusiasts.

🏁 Conclusion

Sifting ExoS demonstrates how AI can accelerate exoplanet discovery and public engagement with space data.
By combining machine learning and NASA’s open datasets, the project makes space exploration more interactive, educational, and accessible to all.

*Note: This readme file was refined using AI*
