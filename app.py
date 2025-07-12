import streamlit as st
import pandas as pd
import forecast
from agent import db, terms, chain
import visuals

st.set_page_config(page_title="Hyderabad Rainfall Forecast", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #ebd1b5;
        background-color: #000000;
        background-image: linear-gradient(315deg, #000000 0%, #5e5368 74%);
        background-attachment: fixed;
        background-size: cover;
    }

    input, textarea {
        background-color: #fdfaf6;
        color: #333333;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #cccccc;
        font-size: 14px;
    }

    button, .stButton > button {
        background-color: #507bd9;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }   

    button:hover, .stButton > button:hover {
        background-color: #3e68c2;
        color: beige;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.header("Hyderabad Rainfall Forecast")
df = pd.read_csv("hyd-monthly-rains.csv")
st.write("📊 **Data Preview:**")
st.dataframe(df.head())

monthly_cols = ["Jan", "Feb", "Mar", "April", "May", "June",
                "July", "Aug", "Sept", "Oct", "Nov", "Dec"]


user_year = st.number_input(
    'Enter Target Forecast Year:',
    min_value=2022,
    step=1
)


if st.button("Get Forecast"):
    try:
        fig = forecast.sarima_forecast(df, monthly_cols, user_year)
        st.session_state['forecast_fig'] = fig
        st.warning("⚠️ The predicted values are based on historical averages and do not include uncertainty in rainfall spells.")
    except Exception as e:
        st.error(f"Error: {e}")


if 'forecast_fig' in st.session_state:
    st.plotly_chart(st.session_state['forecast_fig'])

if 'forecast_fig' in st.session_state:
    selected_term = st.selectbox("Pick a term to learn more:", terms)

    if st.button("🔍 Explain It"):
     with st.spinner("Finding best context..."):
        retrieved_docs = db.similarity_search(selected_term, k=1)
        context = retrieved_docs[0].page_content

        result = chain.invoke({"context": context})

        st.success("✅ Explanation generated:")
        st.markdown(result.content, unsafe_allow_html=True)

if st.checkbox('Click to get More Analysis'):
    try:
        avg_df = visuals.average_monthly_rainfall(df)
        st.write("Average Monthly Rainfall (1901–2021)")
        st.table(avg_df)

        user_yr = st.radio(
            "Select the time span:",
            [5, 10, 20],
            format_func=lambda x: {5: "Half Decade (5 years)", 10: "Decade (10 years)", 20: "Two Decades (20 years)"}[x]
        )

        wettest_period, wettest_value, driest_period, driest_value, avg = visuals.get_wettest_and_driest(df, user_yr)

        st.write(f"🌧️ **Wettest period:** {wettest_period} — Avg Rainfall: {wettest_value:.2f} mm")
        st.write(f"🌤️ **Driest period:** {driest_period} — Avg Rainfall: {driest_value:.2f} mm")

        visuals.call_visuals(df)

    except Exception as e:
        st.error(f'Error: {e}')
