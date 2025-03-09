import pandas as pd
import streamlit as st
import pickle
import numpy as np


model = pickle.load(open("model.pkl", "rb"))

def prediction(lenght,beam,hp,age,engine,condition,fiber):
    lenght_beam = lenght * beam
    hp_per_lb = hp / lenght_beam
    hp_per_engine = 0 if engine == 0 else hp / engine

    user = np.array([lenght,beam,engine,hp,lenght_beam,hp_per_lb,hp_per_engine,age,condition,fiber])
    user = user.reshape(1,-1)
    result = model.predict(user)

    return round(result[0],4)

def main():
    st.title("Boat Price Predictor")

    col1, col2 = st.columns(2)

    with col1:
        lenght_ft = st.slider("Lenght (ft)", 1, 100)

        beam_ft = st.slider("Beam (ft)", 1, 250)

        total_hp = st.slider("Total HP", 0, 1600)

        boat_age = st.slider("Boat Age", 0, 85)

    with col2:
        num_engines = int(st.selectbox("Number of Engines", [0, 1, 2, 3, 4]))

        condition = st.radio("Condition", ["New", "Used"])
        condition = 1 if condition == "Used" else 0

        fiberglass = st.radio("Is Huel Material Fiberglass", ["Yes", "No"])
        fiberglass = 1 if condition == "Yes" else 0

    if col2.button("Predict", use_container_width=True):
        result = prediction(lenght_ft,beam_ft,total_hp,boat_age,num_engines,condition,fiberglass)
        st.success(f"Price of the boat: ${result}")

if __name__ == "__main__":
    main()