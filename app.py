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
import os
import hashlib

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Dashboard crimes Montréal", layout="wide")
st.title("🔍 Prédiction des crimes à Montréal")

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

# Pour l'exploration : on applique les filtres de date
filtered_df_exploration = crime_data[
    (crime_data['DATE']>=pd.to_datetime(start_date)) & 
    (crime_data['DATE']<=pd.to_datetime(end_date))
] if apply_period else crime_data.copy()

if filtered_df_exploration.empty:
    st.warning("⚠️ Pas de données pour cette période.")
    st.stop()

types_crime = filtered_df_exploration['CATEGORIE'].unique().tolist()
selected_types = st.sidebar.multiselect("Type de crime", types_crime, default=types_crime)

pdq_options = sorted(filtered_df_exploration['PDQ'].unique())
selected_pdqs = st.sidebar.multiselect("Sélectionner PDQ(s)", pdq_options, default=pdq_options)

show_points = st.sidebar.checkbox("Afficher points d'intersection", value=True)

filtered_df_exploration = filtered_df_exploration[
    (filtered_df_exploration['CATEGORIE'].isin(selected_types)) &
    (filtered_df_exploration['PDQ'].isin(selected_pdqs))
]

if filtered_df_exploration.empty:
    st.warning("⚠️ Pas de données pour ces filtres.")
    st.stop()

# Pour les prévisions : on utilise toujours tout l'historique, sans filtre de date
filtered_df = crime_data[
    (crime_data['CATEGORIE'].isin(selected_types)) &
    (crime_data['PDQ'].isin(selected_pdqs))
]

if filtered_df.empty:
    st.warning("⚠️ Pas de données pour ces filtres.")
    st.stop()

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Exploration", "📈 Prévisions", "📋 Scores MAE/RMSE"])

# -------------------------
# Onglet Exploration
# -------------------------
with tab1:
    st.subheader("Statistiques générales")
    st.write(f"Nombre total d'événements : {len(filtered_df_exploration)}")
    st.write(f"Nombre de PDQ ( postes de quartier) : {filtered_df_exploration['PDQ'].nunique()}")

    filtered_df_exploration['YEAR'] = filtered_df_exploration['DATE'].dt.year
    summary_year = filtered_df_exploration.groupby(['YEAR','CATEGORIE']).size().reset_index(name='Count')
    fig_year = px.bar(summary_year, x='YEAR', y='Count', color='CATEGORIE', barmode='stack',
                      title="Crimes par année et type")
    st.plotly_chart(fig_year, use_container_width=True)

    top_pdq = filtered_df_exploration.groupby('PDQ').size().reset_index(name='Count').sort_values('Count', ascending=False)
    st.write(f"🏆 Le poste de quartier qui recense le plus de crimes - PDQ : {top_pdq.iloc[0]['PDQ']} ({top_pdq.iloc[0]['Count']} crimes)")

    summary_table = filtered_df_exploration.groupby(['PDQ','CATEGORIE']).size().reset_index(name='Count')
    st.dataframe(summary_table)

    st.subheader("Carte des postes de quartier PDQ et crimes")
    with st.spinner("🗺️ Génération de la carte…"):
       
        m = folium.Map(
            location=[pdq_boundaries.geometry.centroid.y.mean(), pdq_boundaries.geometry.centroid.x.mean()],
            zoom_start=11
        )

        for _, row in pdq_boundaries.iterrows():
            pdq_count = filtered_df_exploration[filtered_df_exploration['PDQ']==row['PDQ']].shape[0]
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
            points_count = filtered_df_exploration.groupby(['LATITUDE','LONGITUDE','CATEGORIE']).size().reset_index(name='Count')
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
PREDICTIONS_PROPHET_PATH = "predictions_prophet.csv"
PREDICTIONS_RF_PATH = "predictions_rf.csv"

def save_forecasts(df_prophet, df_rf, start_date, end_date, pdqs, types):
    prophet_file, rf_file = get_forecast_filenames(start_date, end_date, pdqs, types)
    df_prophet.to_csv(prophet_file, index=False)
    df_rf.to_csv(rf_file, index=False)

def load_forecasts(start_date, end_date, pdqs, types):
    prophet_file, rf_file = get_forecast_filenames(start_date, end_date, pdqs, types)
    if os.path.exists(prophet_file) and os.path.exists(rf_file):
        df_prophet = pd.read_csv(prophet_file, parse_dates=['ds'])
        df_rf = pd.read_csv(rf_file, parse_dates=['ds'])
        return df_prophet, df_rf
    return None, None

def get_forecast_filenames(start_date, end_date, pdqs, types):
    key = f"{start_date}_{end_date}_{'_'.join(map(str, sorted(pdqs)))}_{'_'.join(sorted(types))}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    prophet_file = f"predictions_prophet_{hash_key}.csv"
    rf_file = f"predictions_rf_{hash_key}.csv"
    return prophet_file, rf_file

with tab2:
    st.subheader("Prévisions : Prophet vs Random Forest")

    # Filtres spécifiques à la carte
    st.markdown("### 🎛️ Filtres carte")
    col1, col2, col3 = st.columns(3)
    with col1:
        periods = st.slider("Période de prévision (jours)", 7, 90, 30)
    with col2:
        filter_types = st.multiselect("Types de crimes à afficher", selected_types, default=selected_types)
    with col3:
        seuil = st.number_input("Afficher seulement si prévision ≥", min_value=0, value=0)

    refresh = st.button("🔄 Rafraîchir les prévisions (recalculer)")

    # Charger les prévisions sauvegardées si elles existent et pas de refresh demandé
    if not refresh:
        df_prophet_loaded, df_rf_loaded = load_forecasts(start_date, end_date, selected_pdqs, selected_types)
        if df_prophet_loaded is not None and df_rf_loaded is not None:
            st.session_state.df_forecast_prophet = df_prophet_loaded
            st.session_state.df_forecast_rf = df_rf_loaded

    # Calculer et sauvegarder si besoin
    if ("df_forecast_prophet" not in st.session_state or
        "df_forecast_rf" not in st.session_state or
        refresh):

        forecasts_prophet = []
        forecasts_rf = []

        with st.status("Prévisions en cours...", expanded=True) as status:
            total_tasks = len(selected_pdqs) * len(selected_types)
            task_count = 0
            bar = st.progress(0)

            for pdq in selected_pdqs:
                for crime_type in selected_types:
                    task_count += 1
                    st.write(f"🔄 PDQ {pdq}, type {crime_type} , tâche: ({task_count}/{total_tasks})")
                    bar.progress(task_count / total_tasks)

                    df_subset = filtered_df[(filtered_df['PDQ']==pdq) & (filtered_df['CATEGORIE']==crime_type)]
                    if len(df_subset) < 3:
                        continue

                    df_daily = df_subset.groupby('DATE').size().reset_index(name='y').rename(columns={'DATE':'ds'})
                    if len(df_daily) < 2:
                        continue

                    # Prophet
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.1)
                    try:
                        m.fit(df_daily)
                        future = m.make_future_dataframe(periods=periods)
                        future = future[future['ds'] >= pd.to_datetime(datetime.today().date())]
                        if not future.empty:
                            forecast = m.predict(future)
                            forecast['PDQ'] = pdq
                            forecast['CATEGORIE'] = crime_type
                            forecasts_prophet.append(forecast[['ds','yhat','PDQ','CATEGORIE']])
                    except Exception as e:
                        st.warning(f"⚠️ Prophet a échoué pour PDQ {pdq}, {crime_type}: {e}")

                    # Random Forest
                    df_daily['dayofyear'] = df_daily['ds'].dt.dayofyear
                    df_daily['month'] = df_daily['ds'].dt.month
                    df_daily['weekday'] = df_daily['ds'].dt.weekday
                    X = df_daily[['dayofyear','month','weekday']]
                    y = df_daily['y']

                    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
                    rf.fit(X, y)

                    future_rf = pd.DataFrame({'ds': pd.date_range(df_daily['ds'].max() + timedelta(days=1), periods=periods)})
                    future_rf['dayofyear'] = future_rf['ds'].dt.dayofyear
                    future_rf['month'] = future_rf['ds'].dt.month
                    future_rf['weekday'] = future_rf['ds'].dt.weekday
                    future_rf['yhat'] = rf.predict(future_rf[['dayofyear','month','weekday']])
                    future_rf['PDQ'] = pdq
                    future_rf['CATEGORIE'] = crime_type
                    forecasts_rf.append(future_rf[['ds','yhat','PDQ','CATEGORIE']])

            status.update(label="✅ Prévisions terminées", state="complete")

        if not forecasts_prophet or not forecasts_rf:
            st.warning("Pas assez de données pour générer des prévisions.")
            st.stop()

        st.session_state.df_forecast_prophet = pd.concat(forecasts_prophet).drop_duplicates(subset=['ds','PDQ','CATEGORIE'])
        st.session_state.df_forecast_rf = pd.concat(forecasts_rf).drop_duplicates(subset=['ds','PDQ','CATEGORIE'])

        # Sauvegarder les prévisions pour les futurs chargements
        save_forecasts(st.session_state.df_forecast_prophet, st.session_state.df_forecast_rf, start_date, end_date, selected_pdqs, selected_types)

    # Affichage de la carte (cumul sur la période de prévision)
    if "df_forecast_prophet" in st.session_state and "df_forecast_rf" in st.session_state:
        df_forecast_prophet = st.session_state.df_forecast_prophet
        df_forecast_rf = st.session_state.df_forecast_rf

        st.subheader("Carte des prévisions cumulées sur la période")
        # Définir la période de prévision
        min_pred_date = df_forecast_prophet['ds'].min().date()
        max_pred_date = (min_pred_date + timedelta(days=periods-1))

        st.write(f"Période de prévision : {min_pred_date} au {max_pred_date}")

        # Filtrer les prévisions sur la période
        mask_period = (df_forecast_prophet['ds'].dt.date >= min_pred_date) & (df_forecast_prophet['ds'].dt.date <= max_pred_date)
        prophet_period = df_forecast_prophet[mask_period]
        rf_period = df_forecast_rf[(df_forecast_rf['ds'].dt.date >= min_pred_date) & (df_forecast_rf['ds'].dt.date <= max_pred_date)]

        map_period = folium.Map(
            location=[pdq_boundaries.geometry.centroid.y.mean(), pdq_boundaries.geometry.centroid.x.mean()],
            zoom_start=11
        )

        # --- Ajout des polygones PDQ ---
        for _, row in pdq_boundaries.iterrows():
            pdq_id = row['PDQ']
            # Calcul du cumul total des prévisions pour ce PDQ (tous types sélectionnés)
            prophet_sum_total = prophet_period[prophet_period['PDQ'] == pdq_id]['yhat'].sum()
            rf_sum_total = rf_period[rf_period['PDQ'] == pdq_id]['yhat'].sum()
            # Couleur selon la somme Prophet (plus la valeur est haute, plus c'est rouge)
            if prophet_sum_total > 0:
                fill_color = "#FF9999"
            else:
                fill_color = "#CCCCCC"
            folium.GeoJson(
                row['geometry'],
                style_function=lambda feature, clr=fill_color: {
                    'fillColor': clr,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.3,
                },
                tooltip=f"PDQ: {pdq_id} | Prophet (cumul): {prophet_sum_total:.1f} | RF (cumul): {rf_sum_total:.1f}"
            ).add_to(map_period)

        # --- Ajout des points de prévision cumulée par type ---
        for _, row in pdq_boundaries.iterrows():
            pdq_id = row['PDQ']
            crimes_here = [t for t in filter_types if t in selected_types]
            polygon = row['geometry']

            for i, crime_type in enumerate(crimes_here):
                hist_df = filtered_df[
                    (filtered_df['PDQ']==pdq_id) & (filtered_df['CATEGORIE']==crime_type)
                ]
                rolling_hist = hist_df.groupby('DATE').size().rolling(window=periods).sum().dropna()
                hist_mean = rolling_hist.mean() if not rolling_hist.empty else 0

                prophet_sum = prophet_period[
                    (prophet_period['PDQ']==pdq_id) &
                    (prophet_period['CATEGORIE']==crime_type)
                ]['yhat'].sum()

                rf_sum = rf_period[
                    (rf_period['PDQ']==pdq_id) &
                    (rf_period['CATEGORIE']==crime_type)
                ]['yhat'].sum()

                if np.isnan(prophet_sum) or np.isnan(rf_sum) or (prophet_sum<seuil and rf_sum<seuil):
                    continue

                if prophet_sum > hist_mean * 1.1:
                    color = "red"; tendance = "⚠️ Hausse attendue"
                elif prophet_sum < hist_mean * 0.9:
                    color = "green"; tendance = "✅ Baisse attendue"
                else:
                    color = "gray"; tendance = "➖ Stable"

                # 1. Essayer d'utiliser les vraies coordonnées des crimes historiques
                lat, lon = None, None
                if not hist_df.empty:
                    lat = hist_df['LATITUDE'].median()
                    lon = hist_df['LONGITUDE'].median()
                    from shapely.geometry import Point
                    if not polygon.contains(Point(lon, lat)):
                        lat, lon = None, None

                # 2. Sinon, répartir dans le polygone autour du centroïde
                if lat is None or lon is None:
                    centroid = polygon.centroid
                    type_offsets = np.linspace(0, 2 * np.pi, num=len(crimes_here), endpoint=False)
                    offset_radius = 0.002
                    angle = type_offsets[i]
                    lat_offset = offset_radius * np.cos(angle)
                    lon_offset = offset_radius * np.sin(angle)
                    lat = centroid.y + lat_offset
                    lon = centroid.x + lon_offset
                    if not polygon.contains(Point(lon, lat)):
                        lat = centroid.y
                        lon = centroid.x

                popup_text = f"""
                <b>📍 PDQ {pdq_id}</b><br>
                🔎 {crime_type}<br>
                🕒 {min_pred_date} au {max_pred_date}<br><br>
                ✅ Moyenne historique (période équivalente) : {hist_mean:.1f}<br>
                🔵 Prophet (cumul) : {prophet_sum:.1f}<br>
                🔴 RF (cumul) : {rf_sum:.1f}<br><br>
                <b>{tendance}</b>
                """

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=7,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(map_period)

        folium_static(map_period, width=1200, height=600)

        # Ajout d'une légende claire sous la carte
        st.markdown("""
        <div style="background-color:#f9f9f9;padding:10px;border-radius:8px;border:1px solid #eee;">
        <b>🗺️ Légende de la carte :</b><br>
        <ul>
        <li><b>Polygones</b> : PDQ, couleur selon présence de prévision (rouge = prévision, gris = aucune)</li>
        <li><b>Points</b> : prévision cumulée par type de crime (répartis dans le polygone ou selon la médiane des faits historiques)</li>
        <li><b>Couleur du point</b> :<br>
            <span style="color:red;">Rouge</span> = hausse attendue<br>
            <span style="color:green;">Vert</span> = baisse attendue<br>
            <span style="color:gray;">Gris</span> = stable
        </li>
        <li><b>Popup</b> : détail des prévisions Prophet et Random Forest, tendance vs historique</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Fonctions pour sauvegarder/charger les scores
# -------------------------
SCORES_PATH = "scores_mae_rmse.csv"

def save_scores(df_scores):
    df_scores.to_csv(SCORES_PATH, index=False)

def load_scores():
    if os.path.exists(SCORES_PATH):
        return pd.read_csv(SCORES_PATH)
    return None
from prophet.diagnostics import cross_validation, performance_metrics

# -------------------------
# Onglet Scores MAE/RMSE avec cross-validation
# -------------------------
with tab3:
    st.subheader("Comparaison des modèles : MAE et RMSE sur l'historique (Hold-out)")

    run_scores = st.button("🚦 Lancer l'analyse des scores (Hold-out)")

    if run_scores:
        with st.spinner("Calcul des scores en cours..."):
            scores = []
            all_pdqs = sorted(crime_data['PDQ'].unique())
            all_types = sorted(crime_data['CATEGORIE'].unique())
            for pdq in all_pdqs:
                for crime_type in all_types:
                    try:
                        df_subset = crime_data[(crime_data['PDQ'] == pdq) & (crime_data['CATEGORIE'] == crime_type)]
                        if len(df_subset) < 20:
                            st.write(f"PDQ {pdq} - {crime_type} : moins de 20 événements ({len(df_subset)})")
                            continue
                        df_daily = (
                            df_subset.groupby('DATE').size()
                            .reset_index(name='y')
                            .rename(columns={'DATE': 'ds'})
                            .sort_values('ds')
                        )
                        df_daily['ds'] = pd.to_datetime(df_daily['ds'])
                        if len(df_daily) < 20:
                            st.write(f"PDQ {pdq} - {crime_type} : moins de 20 jours distincts ({len(df_daily)})")
                            continue
                        if df_daily['ds'].isnull().any() or df_daily['y'].isnull().any():
                            st.write(f"PDQ {pdq} - {crime_type} : valeurs manquantes dans ds ou y")
                            continue
                        split_idx = int(len(df_daily) * 0.8)
                        train = df_daily.iloc[:split_idx]
                        test = df_daily.iloc[split_idx:]
                        if test.empty or train.empty:
                            st.write(f"PDQ {pdq} - {crime_type} : train={len(train)}, test={len(test)} (test ou train vide)")
                            continue

                        # Prophet
                        try:
                            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.1)
                            m.fit(train)
                            future = test[['ds']]
                            forecast = m.predict(future)
                            yhat_prophet = forecast['yhat'].values
                            yhat_prophet = np.clip(yhat_prophet, 0, None)
                            mae_prophet = mean_absolute_error(test['y'], yhat_prophet)
                            rmse_prophet = np.sqrt(mean_squared_error(test['y'], yhat_prophet))
                        except Exception as e:
                            st.warning(f"Prophet fail: {e}")
                            mae_prophet = rmse_prophet = np.nan

                        # Random Forest
                        try:
                            train_rf = train.copy()
                            test_rf = test.copy()
                            for df_ in [train_rf, test_rf]:
                                df_['dayofyear'] = df_['ds'].dt.dayofyear
                                df_['month'] = df_['ds'].dt.month
                                df_['weekday'] = df_['ds'].dt.weekday
                                df_['lag_1'] = df_['y'].shift(1).fillna(0)
                                df_['lag_7'] = df_['y'].shift(7).fillna(0)
                                df_['rolling_7'] = df_['y'].rolling(7, min_periods=1).mean()
                            X_train = train_rf[['dayofyear','month','weekday','lag_1','lag_7','rolling_7']]
                            y_train = train_rf['y']
                            X_test = test_rf[['dayofyear','month','weekday','lag_1','lag_7','rolling_7']]
                            y_test = test_rf['y']
                            if len(X_test) == 0 or len(y_test) == 0:
                                st.write(f"PDQ {pdq} - {crime_type} : X_test ou y_test vide")
                                mae_rf = rmse_rf = np.nan
                            else:
                                rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
                                rf.fit(X_train, y_train)
                                yhat_rf = rf.predict(X_test)
                                yhat_rf = np.clip(yhat_rf, 0, None)
                                mae_rf = mean_absolute_error(y_test, yhat_rf)
                                rmse_rf = np.sqrt(mean_squared_error(y_test, yhat_rf))
                        except Exception as e:
                            st.warning(f"RF fail: {e}")
                            mae_rf = rmse_rf = np.nan

                        scores.append({
                            "PDQ": pdq,
                            "CATEGORIE": crime_type,
                            "MAE_Prophet": mae_prophet,
                            "RMSE_Prophet": rmse_prophet,
                            "MAE_RF": mae_rf,
                            "RMSE_RF": rmse_rf
                        })
                    except Exception as e:
                        st.warning(f"Erreur pour PDQ {pdq}, {crime_type}: {e}")

            df_scores = pd.DataFrame(scores)
            if not df_scores.empty:
                st.dataframe(df_scores, use_container_width=True)
                st.markdown("""
                <small>
                <b>MAE</b> : Erreur absolue moyenne (plus c'est bas, mieux c'est)<br>
                <b>RMSE</b> : Racine de l'erreur quadratique moyenne (sensible aux grosses erreurs)<br>
                Scores calculés sur 20% des données historiques (test set).<br>
                Les prédictions négatives sont ramenées à zéro.
                </small>
                """, unsafe_allow_html=True)
                # Moyenne globale pour chaque modèle
                mean_mae_prophet = df_scores["MAE_Prophet"].mean()
                mean_mae_rf = df_scores["MAE_RF"].mean()
                mean_rmse_prophet = df_scores["RMSE_Prophet"].mean()
                mean_rmse_rf = df_scores["RMSE_RF"].mean()
                st.write(f"**MAE moyen Prophet :** {mean_mae_prophet:.3f} | **Random Forest :** {mean_mae_rf:.3f}")
                st.write(f"**RMSE moyen Prophet :** {mean_rmse_prophet:.3f} | **Random Forest :** {mean_rmse_rf:.3f}")
            else:
                st.info("Pas assez de données pour afficher les scores.")
    else:
        st.info("Clique sur le bouton ci-dessus pour lancer l'analyse des scores MAE/RMSE.")

