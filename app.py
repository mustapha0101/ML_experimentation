import streamlit as st
import pandas as pd
import geopandas as gpd
from prophet import Prophet
import folium
from streamlit_folium import folium_static
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from datetime import datetime, timedelta
from streamlit_lottie import st_lottie
import requests

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Dashboard crimes Montréal", layout="wide")
st.title("🔍 Dashboard pédagogique des crimes à Montréal")

# -------------------------
# Lottie animations
# -------------------------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")
lottie_predict = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")

# -------------------------
# Chargement des données
# -------------------------
@st.cache_data
def load_pdq_boundaries():
    return gpd.read_file("limitespdq.geojson")

@st.cache_data
def load_crime_data():
    df = pd.read_csv("actes-criminels.csv")
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df.dropna(subset=['DATE', 'LATITUDE', 'LONGITUDE'], inplace=True)
    df['PDQ'] = df['PDQ'].astype(int)
    return df

crime_data = load_crime_data()
pdq_boundaries = load_pdq_boundaries()

# -------------------------
# Sidebar - filtres
# -------------------------
st.sidebar.header("Filtres")
min_date = crime_data['DATE'].min()
max_date = crime_data['DATE'].max()

with st.sidebar.form("periode_form"):
    start_date = st.date_input("Date de début", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("Date de fin", min_value=min_date, max_value=max_date, value=max_date)
    apply_period = st.form_submit_button("Appliquer la période")

filtered_df = crime_data[
    (crime_data['DATE']>=pd.to_datetime(start_date)) & 
    (crime_data['DATE']<=pd.to_datetime(end_date))
] if apply_period else crime_data.copy()

if filtered_df.empty:
    st.warning("⚠️ Pas de données pour cette période.")
    st.stop()

types_crime = filtered_df['CATEGORIE'].unique().tolist()
selected_types = st.sidebar.multiselect("Type de crime", types_crime, default=types_crime)

pdq_options = sorted(filtered_df['PDQ'].unique())
selected_pdqs = st.sidebar.multiselect("Sélectionner PDQ(s)", pdq_options, default=pdq_options)

show_points = st.sidebar.checkbox("Afficher points d'intersection", value=True)

filtered_df = filtered_df[
    (filtered_df['CATEGORIE'].isin(selected_types)) &
    (filtered_df['PDQ'].isin(selected_pdqs))
]

if filtered_df.empty:
    st.warning("⚠️ Pas de données pour ces filtres.")
    st.stop()

# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["📊 Exploration", "📈 Prévisions"])

# -------------------------
# Onglet Exploration
# -------------------------
with tab1:
    st.subheader("Statistiques générales")
    st.write(f"Nombre total d'événements : {len(filtered_df)}")
    st.write(f"Nombre de PDQ : {filtered_df['PDQ'].nunique()}")

    filtered_df['YEAR'] = filtered_df['DATE'].dt.year
    summary_year = filtered_df.groupby(['YEAR','CATEGORIE']).size().reset_index(name='Count')
    fig_year = px.bar(summary_year, x='YEAR', y='Count', color='CATEGORIE', barmode='stack',
                      title="Crimes par année et type")
    st.plotly_chart(fig_year, use_container_width=True)

    top_pdq = filtered_df.groupby('PDQ').size().reset_index(name='Count').sort_values('Count', ascending=False)
    st.write(f"🏆 Top PDQ : {top_pdq.iloc[0]['PDQ']} ({top_pdq.iloc[0]['Count']} crimes)")

    summary_table = filtered_df.groupby(['PDQ','CATEGORIE']).size().reset_index(name='Count')
    st.dataframe(summary_table)

    st.subheader("Carte des PDQ et crimes")
    m = folium.Map(
        location=[pdq_boundaries.geometry.centroid.y.mean(), pdq_boundaries.geometry.centroid.x.mean()],
        zoom_start=11
    )

    for _, row in pdq_boundaries.iterrows():
        pdq_count = filtered_df[filtered_df['PDQ']==row['PDQ']].shape[0]
        folium.GeoJson(
            row['geometry'],
            style_function=lambda feature, clr=('#FF9999' if pdq_count>0 else '#CCCCCC'): {
                'fillColor': clr,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.3,
            },
            tooltip=f"PDQ: {row['PDQ']} | Nombre de crimes: {pdq_count}"
        ).add_to(m)

    if show_points:
        points_count = filtered_df.groupby(['LATITUDE','LONGITUDE','CATEGORIE']).size().reset_index(name='Count')
        for _, crime in points_count.iterrows():
            folium.CircleMarker(
                location=[crime['LATITUDE'], crime['LONGITUDE']],
                radius=3,
                color='red',
                fill=True,
                fill_opacity=0.7,
                tooltip=f"{crime['CATEGORIE']} ({crime['Count']})"
            ).add_to(m)

    folium_static(m, width=1200, height=600)

# -------------------------
# Onglet Prévisions
# -------------------------
with tab2:
    st.subheader("Prévisions Prophet et Random Forest")
    periods = st.sidebar.slider("Période de prévision (jours)", 7, 90, 30)

    forecasts_prophet = []
    forecasts_rf = []

    if lottie_predict:
        with st.spinner("🔄 Génération des prévisions..."):
            st_lottie(lottie_predict, height=100, key="predicting")

            for pdq in selected_pdqs:
                for crime_type in selected_types:
                    df_subset = filtered_df[(filtered_df['PDQ']==pdq) & (filtered_df['CATEGORIE']==crime_type)]
                    if len(df_subset) < 2:
                        continue
                    df_daily = df_subset.groupby('DATE').size().reset_index(name='y').rename(columns={'DATE':'ds'})
                    if len(df_daily) < 2:
                        continue

                    # Prophet
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                    try:
                        m.fit(df_daily)
                    except:
                        continue
                    future = m.make_future_dataframe(periods=periods)
                    future = future[future['ds'] >= pd.to_datetime(datetime.today().date())]
                    if future.empty:
                        continue
                    forecast = m.predict(future)
                    forecast['PDQ'] = pdq
                    forecast['CATEGORIE'] = crime_type
                    forecasts_prophet.append(forecast[['ds','yhat','PDQ','CATEGORIE']])

                    # Random Forest
                    df_daily['dayofyear'] = df_daily['ds'].dt.dayofyear
                    X = df_daily[['dayofyear']]
                    y = df_daily['y']
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    future_rf = pd.DataFrame({'ds': pd.date_range(df_daily['ds'].max()+timedelta(days=1), periods=periods)})
                    future_rf['dayofyear'] = future_rf['ds'].dt.dayofyear
                    future_rf_pred = rf.predict(future_rf[['dayofyear']])
                    future_rf['yhat'] = future_rf_pred
                    future_rf['PDQ'] = pdq
                    future_rf['CATEGORIE'] = crime_type
                    forecasts_rf.append(future_rf[['ds','yhat','PDQ','CATEGORIE']])

    if not forecasts_prophet or not forecasts_rf:
        st.warning("Pas assez de données pour générer des prévisions.")
        st.stop()

    df_forecast_prophet = pd.concat(forecasts_prophet).drop_duplicates(subset=['ds','PDQ','CATEGORIE'])
    df_forecast_rf = pd.concat(forecasts_rf).drop_duplicates(subset=['ds','PDQ','CATEGORIE'])

    st.subheader("Comparaison des prévisions")
    st.write("✅ Prophet vs Random Forest pour chaque PDQ et type de crime")

    selected_day = st.slider("Sélectionner le jour de prévision", 
                             min_value=df_forecast_prophet['ds'].min().date(), 
                             max_value=df_forecast_prophet['ds'].max().date(), 
                             value=df_forecast_prophet['ds'].min().date())

    map_day = folium.Map(
        location=[pdq_boundaries.geometry.centroid.y.mean(), pdq_boundaries.geometry.centroid.x.mean()],
        zoom_start=11
    )

    # Pour chaque PDQ et type, afficher Prophet et RF
    pdq_type_combinations = pd.MultiIndex.from_product([selected_pdqs, selected_types], names=['PDQ','CATEGORIE']).to_frame(index=False)

    for model_name, df_forecast in zip(['Prophet','Random Forest'], [df_forecast_prophet, df_forecast_rf]):
        filter_day = df_forecast[df_forecast['ds']==pd.to_datetime(selected_day)]
        filter_day_full = pdq_type_combinations.merge(filter_day, on=['PDQ','CATEGORIE'], how='left').fillna(0)

        for _, row in pdq_boundaries.iterrows():
            pdq_id = row['PDQ']
            tooltip_lines = []
            hist_mean = filtered_df[filtered_df['PDQ']==pdq_id].groupby('CATEGORIE').size() / max(1,(filtered_df['DATE'].max()-filtered_df['DATE'].min()).days)
            for _, pred in filter_day_full[filter_day_full['PDQ']==pdq_id].iterrows():
                crime_type = pred['CATEGORIE']
                yhat = pred['yhat']
                mean_hist = hist_mean.get(crime_type, 0)
                trend = "Augmentation" if yhat>mean_hist else "Baisse"
                tooltip_lines.append(f"{crime_type} ({model_name}): {yhat:.1f}, Moyenne: {mean_hist:.1f}, {trend}")

                if show_points:
                    pdq_geom = pdq_boundaries[pdq_boundaries['PDQ']==pdq_id].geometry
                    if not pdq_geom.empty:
                        centroid = pdq_geom.centroid.iloc[0]
                        color = px.colors.qualitative.Dark24[selected_types.index(crime_type) % 24]
                        folium.CircleMarker(
                            location=[centroid.y, centroid.x],
                            radius=5,
                            color=color,
                            fill=True,
                            fill_opacity=0.7,
                            tooltip=f"{crime_type} ({model_name}) Prévision: {yhat:.1f}, Moyenne: {mean_hist:.1f}, {trend}"
                        ).add_to(map_day)

            # Tooltip polygone PDQ
            folium.GeoJson(
                row['geometry'],
                style_function=lambda feature: {'fillColor':'#CCCCCC','color':'black','weight':1,'fillOpacity':0.3},
                tooltip="<br>".join(tooltip_lines)
            ).add_to(map_day)

    folium_static(map_day, width=1200, height=600)
    st.markdown("""
    💡 **Légende carte**  
    - 🔴/🔵 Points : prévisions par type de crime (couleur différente)  
    - Polygones : PDQ avec prévisions Prophet et Random Forest
    """)
