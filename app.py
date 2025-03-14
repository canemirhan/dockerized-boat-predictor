import streamlit as st
import pickle
import numpy as np

# Loading model
model = pickle.load(open("model.pkl", "rb"))

# Takes inputs from user and returns the price of the boat
def prediction(length,beam,hp,age,engine,condition,fiber):
    length_beam = length * beam
    hp_per_lb = hp / length_beam
    hp_per_engine = 0 if engine == 0 else hp / engine

    user = np.array([length,beam,engine,hp,length_beam,hp_per_lb,hp_per_engine,age,condition,fiber])
    user = user.reshape(1,-1)
    result = model.predict(user)

    return round(result[0],4)

# STREAMLIT APP #
def main():
    # Title
    st.title("Boat Price Predictor")

    # Columns of the page
    col1, col2 = st.columns(2)

    # Retrieving data from user
    with col1:
        length_ft = st.slider("Length (ft)", 1, 100)

        beam_ft = st.slider("Beam (ft)", 1, 250)

        total_hp = st.slider("Total HP", 0, 1600)

        boat_age = st.slider("Boat Age", 0, 85)

    with col2:
        num_engines = int(st.selectbox("Number of Engines", [0, 1, 2, 3, 4]))

        condition = st.radio("Condition", ["New", "Used"])
        condition = 1 if condition == "Used" else 0

        fiberglass = st.radio("Is Huel Material Fiberglass", ["Yes", "No"])
        fiberglass = 1 if fiberglass == "Yes" else 0

    # Predict button showing the result
    if col2.button("Predict", use_container_width=True):
        result = prediction(length_ft,beam_ft,total_hp,boat_age,num_engines,condition,fiberglass)
        st.success(f"Price of the boat: ${result}")

# main
if __name__ == "__main__":
    main()