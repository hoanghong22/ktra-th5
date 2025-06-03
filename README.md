# ktra-th5
AR1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
df = pd.read_csv("fx_data.csv")
df['Date'] = df['Date'].apply(lambda x: parser.parse(x).replace(tzinfo=None))
df = df.sort_values('Date')
series = df[['Date', 'USDINR_Close']].dropna()
series.columns = ['Date', 'USDINR_Close']
df.head()
df['USDINR_Close'].plot(title='Tỷ giá USD/INR ban đầu')
plt.show()
series_diff = df['USDINR_Close'].diff().dropna()
series_diff.plot(title='Tỷ giá USD/INR - sai phân bậc 1')
plt.show()
model = AutoReg(series_diff, lags=1, old_names=False)
results = model.fit()
phi = results.params[1]
c = results.params[0]
print("Hệ số hồi quy phi =", phi)
print("Hằng số c =", c)
pred_diff = results.predict(start=1, end=len(series_diff))
first_value = df['USDINR_Close'].iloc[0]
forecast_value = pred_diff.cumsum() + first_value
forecast_value.index = series_diff.index
actual = df['USDINR_Close'].loc[forecast_value.index]

mse = mean_squared_error(actual, forecast_value)
mae = mean_absolute_error(actual, forecast_value)

print(f"MSE (Mean Squared Error): {mse}")
print(f"MAE (Mean Absolute Error): {mae}")
plt.figure(figsize=(12,6))
plt.plot(actual, label='Thực tế', color='blue')
plt.plot(forecast_value, label='AR(1) dự báo', linestyle='--', color='orange')
plt.title('So sánh tỷ giá thực tế và dự báo bằng mô hình AR(1)')
plt.xlabel('Thời gian')
plt.ylabel('Tỷ giá USD/INR')
plt.legend()
plt.grid(True)
plt.show()
