import streamlit as st
import pandas as pd
import fiona
import shapely.geometry as geom
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
    features = []
    with fiona.open("limitespdq.geojson") as src:
        for f in src:
            geom_shape = geom.shape(f["geometry"])
            features.append({
                "PDQ": f["properties"]["PDQ"],
                "geometry": geom_shape
            })
    return pd.DataFrame(features)

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
        location=[pdq_boundaries.geometry.apply(lambda g: g.centroid.y).mean(), 
                  pdq_boundaries.geometry.apply(lambda g: g.centroid.x).mean()],
        zoom_start=11
    )

    for _, row in pdq_boundaries.iterrows():
        pdq_count = filtered_df[filtered_df['PDQ']==row['PDQ']].shape[0]
        folium.GeoJson(
            row['geometry'].__geo_interface__,
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
