import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    with open(BASE_DIR / "starbucks_calories_regressor.pkl", "rb") as f:
        regressor = pickle.load(f)
    with open(BASE_DIR / "starbucks_calories_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    return regressor, classifier

st.set_page_config(page_title="Starbucks Calories Predictor", page_icon="☕", layout="centered")

st.title("☕ Starbucks Calories Predictor")
st.write("Enter the drink's nutrition details to estimate calories and classify it as low or high calorie.")
st.info("This app uses the two Decision Tree models supplied with the project. Predictions are estimates.")

try:
    regressor, classifier = load_models()
except Exception as exc:
    st.error("Model files could not be loaded. Keep both .pkl files in the same folder as app.py.")
    st.exception(exc)
    st.stop()

categories = [
    "Coffee",
    "Frappuccino® Blended Coffee",
    "Frappuccino® Blended Crème",
    "Frappuccino® Light Blended Coffee",
    "Shaken Iced Beverages",
    "Signature Espresso Drinks",
    "Smoothies",
    "Tazo® Tea Drinks",
]

with st.form("prediction_form"):
    st.subheader("Drink information")
    category = st.selectbox("Beverage category", categories)
    c1, c2 = st.columns(2)
    with c1:
        total_fat = st.text_input("Total fat (g)", value="2")
        trans_fat = st.number_input("Trans fat (g)", min_value=0.0, value=0.0, step=0.1)
        saturated_fat = st.number_input("Saturated fat (g)", min_value=0.0, value=0.0, step=0.1)
        sodium = st.number_input("Sodium (mg)", min_value=0.0, value=5.0, step=1.0)
        carbs = st.number_input("Total carbohydrates (g)", min_value=0.0, value=20.0, step=1.0)
        cholesterol = st.number_input("Cholesterol (mg)", min_value=0.0, value=5.0, step=1.0)
    with c2:
        fibre = st.number_input("Dietary fibre (g)", min_value=0.0, value=0.0, step=0.5)
        sugars = st.number_input("Sugars (g)", min_value=0.0, value=15.0, step=1.0)
        protein = st.number_input("Protein (g)", min_value=0.0, value=2.0, step=0.5)
        vitamin_a = st.number_input("Vitamin A (% DV)", min_value=0.0, value=0.0, step=5.0)
        vitamin_c = st.number_input("Vitamin C (% DV)", min_value=0.0, value=0.0, step=5.0)
        calcium = st.number_input("Calcium (% DV)", min_value=0.0, value=0.0, step=5.0)
        iron = st.number_input("Iron (% DV)", min_value=0.0, value=0.0, step=5.0)
    caffeine = st.text_input("Caffeine (mg)", value="75")
    submitted = st.form_submit_button("Predict calories")

if submitted:
    try:
        # The notebook trained on one-hot encoded columns after dropping beverage
        # and beverage_prep. Match that preprocessing and the saved model schema.
        row = pd.DataFrame([{
            "trans_fat_g": trans_fat,
            "saturated_fat_g": saturated_fat,
            "sodium_mg": sodium,
            "total_carbohydrates_g": carbs,
            "cholesterol_mg": cholesterol,
            "dietary_fibre_g": fibre,
            "sugars_g": sugars,
            "protein_g": protein,
            "vitamin_a_%_dv": vitamin_a,
            "vitamin_c_%_dv": vitamin_c,
            "calcium_%_dv": calcium,
            "iron_%_dv": iron,
            "beverage_category": category,
            "total_fat_g": str(total_fat).strip(),
            "caffeine_mg": str(caffeine).strip(),
        }])
        features = pd.get_dummies(row, drop_first=True)
        expected = list(regressor.feature_names_in_)
        features = features.reindex(columns=expected, fill_value=0)

        calorie_estimate = float(regressor.predict(features)[0])
        class_prediction = int(classifier.predict(features)[0])

        st.subheader("Prediction result")
        st.metric("Estimated calories", f"{calorie_estimate:.0f} kcal")
        if class_prediction == 1:
            st.warning("Model classification: High Calorie")
        else:
            st.success("Model classification: Low Calorie")
        st.caption("Classification follows the threshold used in the training notebook (median calories).")
    except Exception as exc:
        st.error("Please check the Total fat and Caffeine values. Enter values in the same format as the training dataset.")
        st.exception(exc)

st.divider()
st.caption("Academic project demonstration — predictions may differ from actual Starbucks nutrition information.")
