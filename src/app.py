import streamlit as st
import pickle
import pandas as pd

with open("/workspaces/streamlit/src/sirtuin6.pkl", "rb") as file:
    artifact = pickle.load(file) #Cargamos en tiempo real nuesto modelo creado

st.title("SIRTUIN6 Cell - Model prediction")
val1 = st.slider("SC-5: ", min_value=0.05, max_value=1.0, value=0.5, step=0.01)
val2 = st.slider("SC-6: ", min_value=1.0, max_value=8.0, value=4.0, step=0.05)
val3 = st.slider("SHBd: ", min_value=0.0, max_value=2.0, value=1.0, step=0.02)
val4 = st.slider("minHaaCH: ", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
val5 = st.slider("maxwHBa: ", min_value=0.0, max_value=2.8, value=1.4, step=0.03)
val6 = st.slider("FMF: ", min_value=0.1, max_value=0.6, value=0.35, step=0.01)
df_predict = pd.DataFrame([[val1,val2,val3,val4,val5,val6]], columns=artifact["predictors"])
result = artifact["model"].predict(df_predict)
predicted_label = artifact["target_encoder"].inverse_transform(result)

st.write(predicted_label)