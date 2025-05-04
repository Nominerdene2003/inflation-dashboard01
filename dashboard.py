import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Инфляцын Таамаглалын Дашбоард (ElasticNet)")

# --- File Upload ---
uploaded_file = st.sidebar.file_uploader("📂 normalized_diplomadata.xlsx оруулна уу", type=["xlsx"])
h = st.sidebar.selectbox("📅 Таамаглах хугацаа (сар)", [3, 6, 9, 12])

if uploaded_file:
    data = pd.read_excel(uploaded_file)
    start_date = pd.to_datetime('2008-07-01')
    data['Date'] = pd.date_range(start=start_date, periods=len(data), freq='MS')

    # --- Seasonal features ---
    winter, spring, summer, fall = [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]
    regions = ['KHAN', 'ZUUN', 'TUV', 'BAR']
    features = []
    for region in regions:
        for season, months in zip(['Winter','Spring','Summer','Fall'], [winter, spring, summer, fall]):
            for var in ['TEMP', 'KHUR']:
                base_col = f'CL_{var}_{region}'
                new_col = f'{season}_{var}_{region}'
                data[new_col] = np.where(data['Date'].dt.month.isin(months), data[base_col], 0)
                features.append(new_col)

    macro_vars = [col for col in data.columns if col not in features + ['Date', 'INF'] and not col.startswith('CL_')]
    features += macro_vars

    x_data = data[features]
    y_data = data['INF']
    train_size = int(0.8 * len(data))

    # --- Train ElasticNet ---
    alpha = 0.1  # эсвэл hyperparameter tuning ашиглаж болно
    l1_ratio = 0.5
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(x_data.iloc[:train_size], y_data.iloc[:train_size])

    # --- Хэрэглэгч input (слайдерээр хувьсагч оруулна) ---
    st.subheader("🔢 Input хувьсагчийн утгуудыг оруулна уу")
    user_input = {}
    for feature in features:
        min_val = float(x_data[feature].min())
        max_val = float(x_data[feature].max())
        default_val = float(x_data[feature].mean())
        user_input[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=default_val)

    input_df = pd.DataFrame([user_input])
    forecast = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("📈 Таамагласан инфляц")
    st.metric(label=f"{h} сарын дараах INF таамаг", value=f"{forecast:.2f}%")

    # --- Coefficients ---
    st.subheader("📌 Хувьсагчдын нөлөөлөл (коэффициент)")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    st.dataframe(coef_df.head(20))

else:
    st.warning("📤 Эхлээд normalized_diplomadata.xlsx файлаа оруулна уу.")
