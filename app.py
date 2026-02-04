import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# -------------------------------
# Title
# -------------------------------
st.title("Titanic Passenger Survival Prediction")

st.write("Enter passenger details to predict survival chances.")

# -------------------------------
# User Inputs
# -------------------------------
pclass = st.slider("Passenger Class", 1, 3, 3)
sex = st.selectbox("Gender", ["male", "female"])
sibsp = st.slider("Number of Siblings / Spouse", 0, 8, 0)
parch = st.slider("Number of Parents / Children", 0, 6, 0)
fare = st.number_input("Ticket Fare", min_value=0.0, step=1.0)
embarked = st.selectbox(
    "Port of Embarkation",
    ["Chebourg", "Queenstown", "Southampton"]
)

# -------------------------------
# Create Input DataFrame
# -------------------------------
data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

# -------------------------------
# Load Model & Encoders
# -------------------------------
model = load_model("model.h5")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Preprocessing
# -------------------------------
# Encode Sex
data["Sex"] = label_encoder.transform(data["Sex"])

# OneHot Encode Embarked
embarked_encoded = onehot_encoder.transform(data[["Embarked"]])
embarked_df = pd.DataFrame(
    embarked_encoded,
    columns=onehot_encoder.get_feature_names_out(["Embarked"])
)

# Combine Data
data = pd.concat([data.drop(columns=["Embarked"]), embarked_df], axis=1)

# Scale Numerical Features
data[["Pclass", "SibSp", "Parch", "Fare"]] = scaler.transform(
    data[["Pclass", "SibSp", "Parch", "Fare"]]
)

# -------------------------------
# Prediction
# -------------------------------

y= model.predict(data)[0][0]
def Chance(y):
    if y > 0.5:
        st.success("Passenger is likely to SURVIVE")
    else:
        st.error("Passenger is NOT likely to survive")

if st.button("Predict Survival Chance"):
    st.write('Probablity of Passenger survival Chance', y)
    st.write(Chance(y))