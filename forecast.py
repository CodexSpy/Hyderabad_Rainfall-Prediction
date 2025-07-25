#Q-9 Rainfal Forecasting usin SARIMA (Seasonal Auto Regressive Integrated Moving Average) -fn()=> SARIMA(p,d,q) * (P,D,Q,s)
'''
Where:

(p,d,q) → same as ARIMA: non-seasonal autoregressive, differencing, moving average.

(P,D,Q,s) → seasonal AR, differencing, MA, and s = seasonal period length.

For monthly rainfall, s = 12 (12 months in a Cycle).

'''
import pandas as pd

import warnings

warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

def sarima_forecast(df, monthly_cols, user_year):

 df=df.reset_index(drop=True)


 ts_rainfall = df.set_index('Year')[monthly_cols].stack()
 ts_rainfall.index = pd.date_range(start='1901-01-01', periods=len(ts_rainfall), freq='ME')

 last_date = ts_rainfall.index[-1]
 last_year = last_date.year

 forecast_months=((user_year- last_year)) *12 -1

 if forecast_months  <=0:
   raise ValueError("Target Year must be after last data year e.g. 2021!!")


 order = (1, 1, 1)
 seasonal_order = (1, 1, 1, 12)

 model = SARIMAX(ts_rainfall, order=order, seasonal_order=seasonal_order)
 model_fit = model.fit()

 #print(model_fit.summary())

 forecast = model_fit.get_forecast(steps=forecast_months)
 forecast_ci = forecast.conf_int()
 forecast_mean=forecast.predicted_mean

 future_dates=pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

 fig = go.Figure()

 print("⚠️ The Predicted Values are based upon previous recorded Normal or Average Values and does not count Uncertinity in Rainfall spells")

 fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast_mean,
    mode='lines+markers+text',
    name='Forecast',
    text=[f"{y:.1f} mm" for y in forecast_mean],
    textposition='top center',
    line=dict(color='cyan')
 ))

 fig.add_trace(go.Scatter(
    x=future_dates.tolist() + future_dates[::-1].tolist(),
    y=forecast_ci.iloc[:, 0].tolist() + forecast_ci.iloc[:, 1][::-1].tolist(),
    fill='toself',
    fillcolor='rgba(150, 255, 255, 0.2)',
    line=dict(color='rgba(200, 255, 255, 0.5)'),
    hoverinfo="skip",
    showlegend=True,
    name='95% CI',
 ))


 fig.update_layout(
    title=f'SARIMA Forecast: Upto December {user_year}',
    xaxis_title='Month',
    yaxis_title='Rainfall(mm)',
    width=1000,
    height=600,
    title_x=0.4
 )

 return fig
