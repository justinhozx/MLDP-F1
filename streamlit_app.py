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
# LOAD LOOKUP TABLES
# --------------------------------------------------
@st.cache_data
def load_lookups():
    drivers = pd.read_csv("drivers.csv")
    constructors = pd.read_csv("constructors.csv")
    races = pd.read_csv("races.csv")
    circuits = pd.read_csv("circuits.csv")
    return drivers, constructors, races, circuits

drivers, constructors, races, circuits = load_lookups()

drivers["driver_name"] = (drivers["forename"].fillna("") + " " + drivers["surname"].fillna("")).str.strip()
DRIVER_ID_TO_NAME = dict(zip(drivers["driverId"], drivers["driver_name"]))
DRIVER_NAME_TO_ID = dict(zip(drivers["driver_name"], drivers["driverId"]))

CONSTRUCTOR_ID_TO_NAME = dict(zip(constructors["constructorId"], constructors["name"]))
CONSTRUCTOR_NAME_TO_ID = dict(zip(constructors["name"], constructors["constructorId"]))

# --------------------------------------------------
# LOAD ENGINEERED DATA (THIS IS THE KEY FIX)
# --------------------------------------------------
@st.cache_data
def load_engineered():
    df_app = pd.read_csv("df_app_2022_2024.csv")
    return df_app

df_app = load_engineered()

# --------------------------------------------------
# FEATURE PIPELINE PRINT (no re-defining feature eng; just show what is used)
# --------------------------------------------------
with st.expander("🧾 Show feature engineering / pipeline used at prediction time"):
    st.text("================ PIPELINE USED FOR PREDICTION ================")
    st.text("1) Raw input features taken from race_df[FEATURE_COLS]")
    st.text(f"   - FEATURE_COLS count: {len(FEATURE_COLS)}")
    st.text(f"   - FEATURE_COLS: {FEATURE_COLS}")

    st.text("\n2) One-hot encoding applied using pd.get_dummies:")
    st.text("   - columns=['driverId', 'constructorId', 'circuitId', 'country']")
    st.text("   - drop_first=True")

    # we can’t know the exact dummy columns before data is built,
    # but we DO know what the model expects:
    st.text("\n3) Column alignment step:")
    st.text("   - Using X_COLUMNS (saved from training)")
    st.text(f"   - fitted column count: {len(X_COLUMNS)}")
    st.text("==============================================================")

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio(
    "Prediction Mode",
    ["Realistic Season", "Unseen / What-If"]
)

# Only allow years that exist in df_app (your trusted years)
available_years = sorted(df_app["year"].dropna().unique().astype(int).tolist(), reverse=True)

year = st.sidebar.selectbox(
    "Select Season Year",
    available_years
)

# Driver dropdown filtered by selected year (KEY REQUEST)
drivers_in_year_ids = sorted(df_app.loc[df_app["year"] == year, "driverId"].dropna().unique().astype(int).tolist())
drivers_in_year_names = sorted([DRIVER_ID_TO_NAME.get(did, f"driverId={did}") for did in drivers_in_year_ids])

driver_name = st.sidebar.selectbox(
    "Select Driver (filtered by year)",
    drivers_in_year_names
)
driver_id = DRIVER_NAME_TO_ID.get(driver_name, None)
if driver_id is None:
    # fallback if name lookup fails
    inv = {DRIVER_ID_TO_NAME.get(did, f"driverId={did}"): did for did in drivers_in_year_ids}
    driver_id = inv.get(driver_name)

# --------------------------------------------------
# HELPER: BUILD ALIGNED X (same logic as your notebook deployment helpers)
# --------------------------------------------------
def make_X(race_df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(
        race_df[FEATURE_COLS].copy(),
        columns=["driverId", "constructorId", "circuitId", "country"],
        drop_first=True
    )
    X = X.reindex(columns=X_COLUMNS, fill_value=0)
    return X

# Points system
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# --------------------------------------------------
# REALISTIC SEASON PREDICTION (USES REAL ENGINEERED ROWS)
# --------------------------------------------------
if mode == "Realistic Season":
    st.subheader("📊 Realistic Season Prediction (uses engineered data)")

    # Show default team for that driver in that year (most common constructorId)
    driver_year_rows = df_app[(df_app["year"] == year) & (df_app["driverId"] == int(driver_id))].copy()
    if not driver_year_rows.empty and "constructorId" in driver_year_rows.columns:
        default_constructor = int(driver_year_rows["constructorId"].mode().iloc[0])
        st.info(f"Default team for {driver_name} in {year}: {CONSTRUCTOR_ID_TO_NAME.get(default_constructor, default_constructor)}")

    if st.button("Predict Season"):
        season_rows = df_app[df_app["year"] == year].copy()

        if season_rows.empty:
            st.warning("No engineered rows found for this season.")
        else:
            all_predictions = []

            # process each raceId in order
            race_order = (
                season_rows[["raceId", "round", "race_name"]]
                .drop_duplicates()
                .sort_values(["round", "raceId"])
            )

            for _, rr in race_order.iterrows():
                race_id = int(rr["raceId"])
                race_name = rr["race_name"] if "race_name" in rr.index else f"Race {race_id}"

                race_df = season_rows[season_rows["raceId"] == race_id].copy()
                if race_df.empty:
                    continue

                X = make_X(race_df)
                race_df["pred_score"] = model.predict(X)

                race_df = race_df.sort_values("pred_score").reset_index(drop=True)
                race_df["pred_position"] = np.arange(1, len(race_df) + 1)
                race_df["race"] = race_name

                all_predictions.append(race_df)

            season_pred = pd.concat(all_predictions, ignore_index=True)

            # Driver-only view
            driver_view = season_pred[season_pred["driverId"] == int(driver_id)].copy()
            st.markdown(f"### {driver_name} — predicted finishes in {year}")
            st.dataframe(driver_view[["race", "pred_position"]], use_container_width=True)

            # Standings + champion (fun)
            season_pred["points"] = season_pred["pred_position"].map(POINTS).fillna(0)
            standings = season_pred.groupby("driverId")["points"].sum().sort_values(ascending=False)

            champion_id = int(standings.index[0])
            champion_name = DRIVER_ID_TO_NAME.get(champion_id, f"driverId={champion_id}")
            champ_pts = float(standings.iloc[0])

            st.success(f"🏆 Predicted Champion: {champion_name} ({champ_pts:.0f} pts)")

            # Optional: show top 10 standings
            top10 = standings.head(10).reset_index()
            top10["driver"] = top10["driverId"].map(lambda x: DRIVER_ID_TO_NAME.get(int(x), f"driverId={x}"))
            top10 = top10[["driver", "points"]]
            st.markdown("### Top 10 predicted standings")
            st.dataframe(top10, use_container_width=True)

# --------------------------------------------------
# UNSEEN / WHAT-IF MODE (still “unseen”: no MAE/accuracy)
# --------------------------------------------------
else:
    st.subheader("🧪 Unseen / What-If Scenario (no MAE / accuracy)")

    constructor_name = st.selectbox("Select Constructor", sorted(CONSTRUCTOR_NAME_TO_ID.keys()))
    constructor_id = int(CONSTRUCTOR_NAME_TO_ID[constructor_name])

    grid_pos = st.slider("Grid Position (for selected driver)", 1, 20, 5)

    selected_tracks = st.multiselect(
        "Select Circuits",
        circuits["name"].tolist(),
        default=circuits["name"].head(5).tolist()
    )

    # Use last available year as template baseline
    template_year = int(df_app["year"].max())

    if st.button("Run What-If Season"):
        st.info("Unseen scenario: predictions only (no MAE / accuracy).")

        results = []

        # helper: pick template race rows for each circuit (so track differences matter)
        def pick_template_for_circuit(cid: int) -> pd.DataFrame:
            cand = df_app[df_app["circuitId"] == cid].copy()
            if not cand.empty:
                # take most recent raceId at that circuit
                # (uses year+round since df_app already contains them)
                last_race = cand.sort_values(["year", "round"]).iloc[-1]["raceId"]
                return df_app[df_app["raceId"] == int(last_race)].copy()

            # fallback: last race in template_year
            tmp = df_app[df_app["year"] == template_year].copy()
            if not tmp.empty:
                last_race = tmp.sort_values(["round", "raceId"]).iloc[-1]["raceId"]
                return df_app[df_app["raceId"] == int(last_race)].copy()

            return df_app.head(30).copy()

        for i, track in enumerate(selected_tracks, start=1):
            c = circuits[circuits["name"] == track].iloc[0]
            cid = int(c["circuitId"])

            race_df = pick_template_for_circuit(cid)

            # overwrite circuit/location fields
            race_df["year"] = year
            race_df["round"] = i
            race_df["circuitId"] = cid
            if "country" in race_df.columns:
                race_df["country"] = c["country"]
            if "alt" in race_df.columns:
                race_df["alt"] = c["alt"]
            if "lat" in race_df.columns:
                race_df["lat"] = c["lat"]
            if "lng" in race_df.columns:
                race_df["lng"] = c["lng"]

            # apply what-if overrides for the selected driver
            race_df.loc[race_df["driverId"] == int(driver_id), "constructorId"] = constructor_id
            race_df.loc[race_df["driverId"] == int(driver_id), "grid"] = float(grid_pos)

            # predict and rank
            X = make_X(race_df)
            race_df["pred_score"] = model.predict(X)
            race_df = race_df.sort_values("pred_score").reset_index(drop=True)
            race_df["pred_position"] = np.arange(1, len(race_df) + 1)

            # grab selected driver
            drow = race_df[race_df["driverId"] == int(driver_id)].iloc[0]
            results.append({
                "Round": i,
                "Track": track,
                "Predicted Position": int(drow["pred_position"])
            })

        st.dataframe(pd.DataFrame(results), use_container_width=True)
