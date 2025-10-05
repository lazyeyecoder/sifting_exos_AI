# app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------------- Load Model ----------------
model = joblib.load('siftingexosaccurate_model.pkl')
classes = ["Confirmed Exoplanet", "Candidate", "False Positive"]

st.set_page_config(page_title="Exoplanet Identifier", layout="centered")

st.title("SiftingExos")
st.write("Turning star winks into discoveries")
st.write("""
This interactive tool uses a trained **Random Forest AI model** based on NASA Kepler data  
to predict whether a celestial signal represents a **Confirmed Exoplanet**, a **Candidate**, or a **False Positive**.
""")

# ---------------- Feature Descriptions + Ranges ----------------
feature_info = {
    "koi_fpflag_ec": ("Eclipsing Binary Flag", "1 if the signal may come from a binary star, 0 otherwise", (0, 1)),
    "koi_model_snr": ("Model Signal-to-Noise Ratio", "Higher SNR → clearer detection signal", (0, 1000)),
    "koi_prad": ("Planet Radius (Earth Radii)", "Size of the planet compared to Earth", (0.1, 30)),
    "koi_impact": ("Impact Parameter", "How central the planet's transit is across the star (0 = center, >1 = grazing)", (0, 2)),
    "koi_fpflag_ss": ("Secondary Eclipse Flag", "1 if a secondary eclipse (false signal) is detected, 0 otherwise", (0, 1)),
    "koi_fpflag_co": ("Centroid Offset Flag", "1 if the signal shifts off-target, suggesting false detection", (0, 1)),
    "koi_duration": ("Transit Duration (Hours)", "How long the planet blocks the star’s light", (0.05, 72)),
    "koi_period": ("Orbital Period (Days)", "Time the planet takes to orbit its star", (0.3, 1000)),
    "koi_depth": ("Transit Depth (ppm)", "How much light the planet blocks; larger = deeper dip", (1, 50000))
}

# ---------------- Helper: Explain Prediction ----------------
def explain_prediction(inputs):
    impact = inputs['koi_impact']
    radius = inputs['koi_prad']
    snr = inputs['koi_model_snr']
    depth = inputs['koi_depth']
    
    reasons = []
    if impact > 1:
        reasons.append("Impact parameter > 1 → likely not an exoplanet (grazing transit)")
    if radius > 10:
        reasons.append("Planet radius unusually large → likely false positive (stellar companion)")
    if snr < 10:
        reasons.append("Low signal-to-noise ratio → detection uncertain")
    if depth > 10000:
        reasons.append("Transit depth too high → may indicate binary or noise source")
    
    if not reasons:
        reasons.append("All parameters within typical exoplanet range → strong candidate/confirmed")
    
    return "; ".join(reasons)

# ---------------- User Inputs ----------------
st.header("Input Observational Parameters")

inputs = {}
col1, col2 = st.columns(2)

# Loop through features neatly in two columns
for i, (feature, (label, description, (min_val, max_val))) in enumerate(feature_info.items()):
    with col1 if i % 2 == 0 else col2:
        st.markdown(f"**{label}**")
        st.caption(description)
        if "fpflag" in feature:
            inputs[feature] = st.number_input(
                f"Value (Range: {min_val}-{max_val})",
                min_value=0, max_value=1, value=0, key=feature
            )
        else:
            default_val = round((min_val + max_val) / 10, 2)
            inputs[feature] = st.number_input(
                f"Value (Range: {min_val}-{max_val})",
                min_value=float(min_val), max_value=float(max_val), value=default_val, key=feature
            )
        st.markdown("---")

# ---------------- Prediction ----------------
if st.button("Predict Exoplanet Type"):
    input_array = np.array([list(inputs.values())])
    
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    
    pred_class = classes[prediction]
    confidence = np.max(probabilities) * 100
    
    st.success(f"Prediction: {pred_class}")
    st.info(f"Model Confidence: {confidence:.2f}%")
    st.write("Reason for Prediction:", explain_prediction(inputs))

    # Show probability chart
    prob_df = pd.DataFrame({
        "Class": classes,
        "Probability (%)": [p*100 for p in probabilities]
    })
    st.bar_chart(prob_df.set_index("Class"))

else:
    st.info("Enter all values above and click **Predict Exoplanet Type** to see results.")

# ---------------- Footer ----------------
st.markdown("""
---
### ℹ️ About This App
- **Developed by:** Aryan Maurya  
- **Trained on:** NASA Exoplanet Archive (Kepler KOI dataset)  
- **Model Used:** Random Forest Classifier  
- The app predicts whether a detected light dip is a real exoplanet or a false signal.
""")
