import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG (F1 THEME)
# --------------------------------------------------
st.set_page_config(
    page_title="F1 Season Predictor",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ Formula 1 Season Predictor")
st.caption("Powered by HistGradientBoostingRegressor (HGB)")

# --------------------------------------------------
# LOAD MODEL & METADATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    X_columns = joblib.load("X_columns.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, X_columns, feature_cols

model, X_COLUMNS, FEATURE_COLS = load_model()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    drivers = pd.read_csv("drivers.csv")
    constructors = pd.read_csv("constructors.csv")
    races = pd.read_csv("races.csv")
    circuits = pd.read_csv("circuits.csv")
    return drivers, constructors, races, circuits

drivers, constructors, races, circuits = load_data()

# Driver name mapping
drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
DRIVER_MAP = dict(zip(drivers["driver_name"], drivers["driverId"]))
DRIVER_MAP_INV = dict(zip(drivers["driverId"], drivers["driver_name"]))

# Constructor mapping
CONSTRUCTOR_MAP = dict(zip(constructors["name"], constructors["constructorId"]))
CONSTRUCTOR_MAP_INV = dict(zip(constructors["constructorId"], constructors["name"]))

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio(
    "Prediction Mode",
    ["Realistic Season", "Unseen / What-If"]
)

driver_name = st.sidebar.selectbox(
    "Select Driver",
    sorted(DRIVER_MAP.keys())
)
driver_id = DRIVER_MAP[driver_name]

year = st.sidebar.selectbox(
    "Select Season Year",
    sorted(races["year"].unique(), reverse=True)
)

# --------------------------------------------------
# HELPER: BUILD ALIGNED X
# --------------------------------------------------
def make_X(race_df):
    X = pd.get_dummies(
        race_df[FEATURE_COLS],
        columns=["driverId", "constructorId", "circuitId", "country"],
        drop_first=True
    )
    X = X.reindex(columns=X_COLUMNS, fill_value=0)
    return X

# --------------------------------------------------
# REALISTIC SEASON PREDICTION
# --------------------------------------------------
if mode == "Realistic Season":
    st.subheader("📊 Realistic Season Prediction")

    season_races = races[races["year"] == year].sort_values("round")

    if season_races.empty:
        st.warning("No race data for this season.")
    else:
        if st.button("Predict Season"):
            all_predictions = []

            for _, r in season_races.iterrows():
                race_id = r["raceId"]
                circuit_id = r["circuitId"]

                race_df = pd.merge(
                    drivers[["driverId"]],
                    pd.DataFrame({"raceId": [race_id]}),
                    how="cross"
                )

                race_df["year"] = year
                race_df["round"] = r["round"]
                race_df["circuitId"] = circuit_id

                # Merge circuit info
                c = circuits[circuits["circuitId"] == circuit_id].iloc[0]
                race_df["country"] = c["country"]
                race_df["alt"] = c["alt"]
                race_df["lat"] = c["lat"]
                race_df["lng"] = c["lng"]

                # Dummy but consistent defaults
                race_df["grid"] = 10
                race_df["has_quali"] = 1
                race_df["quali_position"] = 10
                race_df["quali_best_sec"] = 90

                race_df["driver_age"] = 30
                race_df["driver_races_before"] = 100
                race_df["constructor_races_before"] = 200
                race_df["driver_last_finish"] = 10
                race_df["driver_avg_finish_last5"] = 10
                race_df["driver_avg_points_last5"] = 2
                race_df["constructor_avg_finish_last5"] = 10
                race_df["constructor_avg_points_last5"] = 10
                race_df["driver_season_points_to_date"] = 0
                race_df["constructor_season_points_to_date"] = 0

                # Assign teams based on historical default
                race_df["constructorId"] = constructors["constructorId"].mode()[0]

                X = make_X(race_df)
                race_df["pred_score"] = model.predict(X)

                race_df = race_df.sort_values("pred_score").reset_index(drop=True)
                race_df["pred_position"] = np.arange(1, len(race_df) + 1)

                race_df["race"] = r["name"]
                all_predictions.append(race_df)

            season_df = pd.concat(all_predictions)

            driver_view = season_df[season_df["driverId"] == driver_id]
            st.dataframe(driver_view[["race", "pred_position"]])

            # Points system
            POINTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
            season_df["points"] = season_df["pred_position"].map(POINTS).fillna(0)

            standings = (
                season_df.groupby("driverId")["points"]
                .sum()
                .sort_values(ascending=False)
            )

            champion_id = standings.index[0]
            st.success(f"🏆 Predicted Champion: {DRIVER_MAP_INV[champion_id]}")

# --------------------------------------------------
# UNSEEN / WHAT-IF MODE
# --------------------------------------------------
else:
    st.subheader("🧪 Unseen / What-If Scenario")

    constructor_name = st.selectbox(
        "Select Constructor",
        sorted(CONSTRUCTOR_MAP.keys())
    )
    constructor_id = CONSTRUCTOR_MAP[constructor_name]

    grid_pos = st.slider("Grid Position", 1, 20, 5)

    selected_tracks = st.multiselect(
        "Select Circuits",
        circuits["name"].tolist(),
        default=circuits["name"].head(5).tolist()
    )

    if st.button("Run What-If Season"):
        st.info("No MAE / accuracy shown (unseen scenario).")

        results = []

        for track in selected_tracks:
            c = circuits[circuits["name"] == track].iloc[0]

            race_df = pd.DataFrame({
                "year": [year],
                "round": [1],
                "circuitId": [c["circuitId"]],
                "alt": [c["alt"]],
                "lat": [c["lat"]],
                "lng": [c["lng"]],
                "grid": [grid_pos],
                "has_quali": [1],
                "quali_position": [grid_pos],
                "quali_best_sec": [90],
                "driver_age": [30],
                "driver_races_before": [100],
                "constructor_races_before": [200],
                "driver_last_finish": [10],
                "driver_avg_finish_last5": [10],
                "driver_avg_points_last5": [2],
                "constructor_avg_finish_last5": [10],
                "constructor_avg_points_last5": [10],
                "driver_season_points_to_date": [0],
                "constructor_season_points_to_date": [0],
                "driverId": [driver_id],
                "constructorId": [constructor_id],
                "country": [c["country"]],
            })

            X = make_X(race_df)
            pred = model.predict(X)[0]

            results.append({
                "Track": track,
                "Predicted Position": int(round(pred))
            })

        st.dataframe(pd.DataFrame(results))
