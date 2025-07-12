import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import streamlit as st
import plotly.express as px
import seaborn as sns

warnings.filterwarnings('ignore')

#Q-1 what is the average monthly rainfall

def average_monthly_rainfall(df):
 months_map = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "April": "April",
    "May": "May",
    "June": "June",
    "July": "July",
    "Aug": "August",
    "Sept": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December"
 }
 averages={}
 for col in df.columns:
   if col !='Year' and col!='Total':
       Avg=np.average(df[col])
       averages[months_map[col]]=Avg
       avg_df = pd.DataFrame(list(averages.items()), columns=['Month', 'Average Rainfall'])
 return avg_df
 
#Q-2 Wettest and Driest 
def get_wettest_and_driest(df, user_yr):
    df_usr = df.copy()

    if user_yr == 5:
        df_usr["half_decade"] = (df_usr["Year"] // 5) * 5
        avg = df_usr.groupby("half_decade")["Total"].mean().round(2)
    elif user_yr == 10:
        df_usr["decade"] = (df_usr["Year"] // 10) * 10
        avg = df_usr.groupby("decade")["Total"].mean().round(2)
    else:
        df_usr["two_decade"] = (df_usr["Year"] // 20) * 20
        avg = df_usr.groupby("two_decade")["Total"].mean().round(2)

    wettest_period = avg.idxmax()
    wettest_value = avg.max()

    driest_period = avg.idxmin()
    driest_value = avg.min()

    return wettest_period, wettest_value, driest_period, driest_value, avg
 
def call_visuals(df):
    st.header("Rainfall Analysis & Visualizations")

    monthly_cols = [
        "Jan", "Feb", "Mar", "April", "May",
        "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"
    ]

    st.subheader("Months with Highly Unpredictable Rainfall (High Std Dev)")

    stds = {col: df[col].std() for col in monthly_cols}
    sorted_stds = sorted(stds.items(), key=lambda x: x[1], reverse=True)

    st.write("**Top 5 Standard Deviation by Month:**")
    st.table(pd.DataFrame(sorted_stds, columns=['Month', 'Std Dev']).head(5))

    std_values = [stds[col] for col in monthly_cols]
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(monthly_cols, std_values, color='#2ca02c', marker='o', s=70, edgecolor='black')
    ax.plot(monthly_cols, std_values, linestyle='-', color='#1f77b4', alpha=0.7)
    ax.set_title("Months with High Unpredictability", fontsize=13, weight='bold')
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Std Dev", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig, use_container_width=True)



    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Rainfall Trend Over Years")
        fig1 = px.scatter(
            df,
            x='Year',
            y='Total',
            trendline='ols',
            trendline_color_override='#8e44ad',
            labels={'Total': 'Total Rainfall (mm)'},
            title='Annual Rainfall Trend'
        )
        fig1.update_traces(marker=dict(size=7, color='#2980b9'))
        fig1.update_layout(
            width=600, height=500,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            title_x=0.4
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Trend of Monsoon Onset Over Years")
        monsoon_months = ['June', 'July', 'Aug', 'Sept']

        onset_month = []
        for idx, row in df.iterrows():
            year = row['Year']
            for month in monsoon_months:
                if row[month] > 68:
                    onset_month.append({'Year': year, 'OnsetMonth': month})
                    break

        df_onset = pd.DataFrame(onset_month)
        month_order = {'June': 6, 'July': 7, 'Aug': 8, 'Sept': 9}
        df_onset['NumericOnset'] = df_onset['OnsetMonth'].map(month_order)

        fig2 = px.scatter(
            df_onset,
            x='Year',
            y='NumericOnset',
            trendline='ols',
            trendline_color_override='#e74c3c',
            labels={'NumericOnset': 'Onset Month'},
            title='Monsoon Onset Trend'
        )
        fig2.update_traces(marker=dict(size=7, color='#e67e22'))
        fig2.update_layout(
            width=600, height=500,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            title_x=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("Extreme Rainfall Frequency")
        threshold = {month: df[month].mean() + 2 * df[month].std() for month in monthly_cols}

        ext_cnt = []
        for idx, row in df.iterrows():
            year = row['Year']
            extreme_months = [month for month in monthly_cols if row[month] > threshold[month]]
            ext_cnt.append({'Year': year, 'ExtremeCount': len(extreme_months)})

        df_extreme = pd.DataFrame(ext_cnt)

        fig3 = px.scatter(
            df_extreme,
            x='Year',
            y='ExtremeCount',
            trendline='ols',
            trendline_color_override='yellow',
            labels={'ExtremeCount': 'Extreme Months'},
            title='Extreme Rainfall Frequency'
        )
        fig3.update_traces(marker=dict(size=7, color='#16a085'))
        fig3.update_layout(
            width=600, height=500,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            title_x=0.4
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Decadal Monthly Rainfall Comparison")
        df['decade'] = (df['Year'] // 10) * 10
        decades_to_compare = [1980, 2020]
        subset = df[df['decade'].isin(decades_to_compare)]

        monthly_mean = subset.groupby('decade')[monthly_cols].mean().T
        monthly_mean.columns = [f"{d}s" for d in monthly_mean.columns]

        x = np.arange(len(monthly_cols))
        width = 0.4

        fig4, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, monthly_mean.iloc[:, 0], width=width, label=f"{decades_to_compare[0]}s", color='#3498db')
        ax.bar(x + width, monthly_mean.iloc[:, 1], width=width, label=f"{decades_to_compare[1]}s", color='#f39c12')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(monthly_cols, rotation=45)
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel('Avg Rainfall (mm)', fontsize=11)
        ax.set_title(f'Monthly Rainfall: {decades_to_compare[0]}s vs {decades_to_compare[1]}s', fontsize=15, weight='bold')
        ax.legend()
        sns.despine()
        st.pyplot(fig4, use_container_width=True)

    st.subheader("Annual Rainfall Anomaly")
    ln_mean = df['Total'].mean()
    df['Anomaly'] = df['Total'] - ln_mean
    df['Anomaly_color'] = df['Anomaly'].apply(lambda x: 'Below Normal' if x < 0 else 'Above Normal')

    fig5 = px.bar(
        df,
        x='Year',
        y='Anomaly',
        color='Anomaly_color',
        title="Annual Rainfall Anomaly",
        labels={'Anomaly': 'Yearly Anomaly'},
        color_discrete_map={'Below Normal': '#c0392b', 'Above Normal': '#27ae60'},
    )
    fig5.update_layout(
        width=1200, height=500,
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig5, use_container_width=True)


