import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from googlesearch import search
from bs4 import BeautifulSoup


import pandas as pd
import numpy as np

# Drivers
drivers = pd.read_csv("archive/drivers.csv", na_values=[r"\N"])
drivers["driver_name"] = (drivers["forename"].fillna("") + " " + drivers["surname"].fillna("")).str.strip()
DRIVER_ID_TO_NAME = dict(zip(drivers["driverId"], drivers["driver_name"]))
DRIVER_NAME_TO_ID = dict(zip(drivers["driver_name"], drivers["driverId"]))

# Constructors
constructors = pd.read_csv("archive/constructors.csv", na_values=[r"\N"])
CONSTRUCTOR_ID_TO_NAME = dict(zip(constructors["constructorId"], constructors["name"]))
CONSTRUCTOR_NAME_TO_ID = dict(zip(constructors["name"], constructors["constructorId"]))

# Races
races = pd.read_csv("archive/races.csv", na_values=[r"\N"])
races["race_name"] = races["name"]
RACE_ID_TO_NAME = dict(zip(races["raceId"], races["race_name"]))

# --------------------------------------------------
# Results
# --------------------------------------------------
results = pd.read_csv("archive/results.csv", na_values=[r"\N"])
results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
results["race_name"] = results["raceId"].map(RACE_ID_TO_NAME)
results["driver_name"] = results["driverId"].map(DRIVER_ID_TO_NAME)
results["constructor_name"] = results["constructorId"].map(CONSTRUCTOR_ID_TO_NAME)
# Races
races = pd.read_csv("archive/races.csv", na_values=[r"\N"])
races["race_name"] = races["name"]

# Map raceId -> race name and year
RACE_ID_TO_NAME = dict(zip(races["raceId"], races["race_name"]))
RACE_ID_TO_YEAR = dict(zip(races["raceId"], races["year"]))  # <-- ADD THIS
# <-- ADD THIS LINE -->
results["year"] = results["raceId"].map(RACE_ID_TO_YEAR)

# ==========================================================
# STREAMLIT APP — 2022–2024 ONLY
# Mirrors the notebook pipeline + adds:
#   ✅ team dropdown limited to constructors that appear in chosen year
#   ✅ target round shows track/race name
#   ✅ shows full calendar for chosen year (round -> race name)
#
# CLEAN UI CHANGE (what-if only):
#   ✅ Hide pipeline expander
#   ✅ Hide override debug checkbox + debug prints
#   ✅ In What-If: show only Year / Track / Driver / Team / Prediction (clean card)
#   ✅ Still keeps Realistic Season mode exactly as before
# ==========================================================

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="F1 2022–2024 Predictor",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Predictor (2022–2024)")
st.caption("Uses your engineered dataset + HGB model. Unseen mode applies constructor/car-performance override like your notebook.")

# --------------------------------------------------
# LOAD MODEL & METADATA (FROM NOTEBOOK EXPORTS)
# --------------------------------------------------
@st.cache_resource
def load_model_and_meta():
    model = joblib.load("model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")

    # Most reliable: exact columns model was fitted on
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        expected_cols = joblib.load("X_columns.pkl")

    return model, expected_cols, feature_cols

MODEL, EXPECTED_X_COLS, FEATURE_COLS = load_model_and_meta()

# --------------------------------------------------
# LOAD LOOKUPS
# --------------------------------------------------
@st.cache_data
def load_lookups():
    drivers = pd.read_csv("archive/drivers.csv")
    constructors = pd.read_csv("archive/constructors.csv")
    races = pd.read_csv("archive/races.csv")
    circuits = pd.read_csv("archive/circuits.csv")
    return drivers, constructors, races, circuits

drivers, constructors, races, circuits = load_lookups()

drivers["driver_name"] = (drivers["forename"].fillna("") + " " + drivers["surname"].fillna("")).str.strip()
DRIVER_ID_TO_NAME = dict(zip(drivers["driverId"], drivers["driver_name"]))
DRIVER_NAME_TO_ID = dict(zip(drivers["driver_name"], drivers["driverId"]))

CONSTRUCTOR_ID_TO_NAME = dict(zip(constructors["constructorId"], constructors["name"]))
CONSTRUCTOR_NAME_TO_ID = dict(zip(constructors["name"], constructors["constructorId"]))

RACE_ID_TO_NAME = dict(zip(races["raceId"], races["name"])) if "raceId" in races.columns and "name" in races.columns else {}

# --------------------------------------------------
# LOAD ENGINEERED DATA (YOUR DEPLOYMENT DATASET)
# --------------------------------------------------
@st.cache_data
def load_engineered():
    return pd.read_csv("archive/df_app_2022_2024.csv")

df_app = load_engineered()
df_app = df_app[df_app["year"].between(2022, 2024)].copy()

# Ensure a clean race_name column WITHOUT suffix collisions
if "race_name" in df_app.columns and "raceId" in df_app.columns:
    df_app["race_name"] = df_app["race_name"].fillna(df_app["raceId"].map(RACE_ID_TO_NAME))
else:
    if "raceId" in df_app.columns and "raceId" in races.columns:
        races_small = races[["raceId", "name"]].rename(columns={"name": "race_name"})
        df_app = df_app.merge(races_small, on="raceId", how="left")

if "race_name" not in df_app.columns and ("race_name_x" in df_app.columns or "race_name_y" in df_app.columns):
    df_app["race_name"] = df_app.get("race_name_x")
    if "race_name_y" in df_app.columns:
        df_app["race_name"] = df_app["race_name"].fillna(df_app["race_name_y"])
    df_app = df_app.drop(columns=[c for c in ["race_name_x", "race_name_y"] if c in df_app.columns])

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio(
    "Prediction Mode",
    ["Realistic Season (2022–2024)", "Unseen / What-If (single race)"]
)

available_years = [2024, 2023, 2022]
year = st.sidebar.selectbox("Season Year", available_years, index=0)

# Show full calendar for chosen year (round -> race name)
st.sidebar.subheader("📅 Season Calendar")
cal = (
    df_app[df_app["year"] == year][["round", "raceId", "race_name"]]
    .drop_duplicates()
    .sort_values(["round", "raceId"])
)
if cal.empty:
    st.sidebar.write("No calendar data for this year.")
else:
    cal_view = cal.copy()
    cal_view["label"] = cal_view.apply(lambda r: f"R{int(r['round']):02d} — {r['race_name']}", axis=1)
    st.sidebar.dataframe(cal_view[["label"]], use_container_width=True, height=260)

# Driver list filtered by year
drivers_in_year_ids = sorted(df_app.loc[df_app["year"] == year, "driverId"].dropna().unique().astype(int).tolist())
drivers_in_year_names = sorted([DRIVER_ID_TO_NAME.get(did, f"driverId={did}") for did in drivers_in_year_ids])

driver_name = st.sidebar.selectbox("Driver (filtered by year)", drivers_in_year_names)
driver_id = DRIVER_NAME_TO_ID.get(driver_name, None)
if driver_id is None:
    inv = {DRIVER_ID_TO_NAME.get(did, f"driverId={did}"): did for did in drivers_in_year_ids}
    driver_id = int(inv.get(driver_name))
driver_id = int(driver_id)

# --------------------------------------------------
# HELPERS (MATCH NOTEBOOK)
# --------------------------------------------------
def make_X(race_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in race_df.columns]
    if missing:
        raise ValueError(
            "Engineered race_df is missing columns required by FEATURE_COLS.\n"
            f"Missing: {missing}\n"
            "Fix: export df_app_2022_2024.csv AFTER feature engineering + imputation (from the notebook)."
        )

    X = pd.get_dummies(
        race_df[FEATURE_COLS].copy(),
        columns=["driverId", "constructorId", "circuitId", "country"],
        drop_first=True
    )

    # Align EXACTLY to model expected columns
    X = X.reindex(columns=EXPECTED_X_COLS, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X


def safe_div(a, b, default=0.0):
    a = float(a) if pd.notna(a) else np.nan
    b = float(b) if pd.notna(b) else np.nan
    if pd.isna(a) or pd.isna(b) or b == 0:
        return float(default)
    return float(a / b)


def get_car_perf_cols(feature_cols, df_in):
    explicit = [
        "constructor_races_before",
        "constructor_avg_finish_last5",
        "constructor_avg_points_last5",
        "constructor_season_points_to_date",
        "team_avg_finish_last5_teammates",
        "team_avg_points_last5_teammates",
        "team_avg_finish_last_year_same_track",
        "team_finish_last5_rank",
        "team_points_last5_rank",
        "team_finish_last_year_track_rank",
    ]
    base = [c for c in explicit if c in feature_cols]
    base += [c for c in feature_cols if c.startswith("constructor_")]
    base = [c for c in base if c in df_in.columns]

    seen = set()
    out = []
    for c in base:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def apply_constructor_override_with_car_perf_and_driver_vs_team(
    race_df: pd.DataFrame,
    season_rows: pd.DataFrame,
    driver_id: int,
    new_constructor_id: int,
    round_no: int,
    debug: bool = False  # ✅ default OFF for clean UI
) -> pd.DataFrame:
    out = race_df.copy()
    driver_id = int(driver_id)
    new_constructor_id = int(new_constructor_id)
    round_no = int(round_no)

    car_perf_cols = get_car_perf_cols(FEATURE_COLS, out)

    # (Debug snapshot removed from UI; kept structure for compatibility)
    out.loc[out["driverId"] == driver_id, "constructorId"] = new_constructor_id

    donor_vals = None

    # same race donor
    if car_perf_cols:
        donor_same = out[out["constructorId"] == new_constructor_id].copy()
        if not donor_same.empty:
            donor_vals = donor_same[car_perf_cols].median(numeric_only=True)

    # prior race donor
    if (donor_vals is None) and car_perf_cols:
        pool_prior = season_rows[
            (season_rows["constructorId"] == new_constructor_id) &
            (season_rows["round"] < round_no)
        ].copy()
        if not pool_prior.empty:
            last_race_id = int(pool_prior.sort_values(["round", "raceId"]).iloc[-1]["raceId"])
            donor_race = season_rows[season_rows["raceId"] == last_race_id].copy()
            if not donor_race.empty:
                donor_vals = donor_race[car_perf_cols].median(numeric_only=True)

    # year median donor
    if (donor_vals is None) and car_perf_cols:
        pool_year = season_rows[season_rows["constructorId"] == new_constructor_id].copy()
        if not pool_year.empty:
            donor_vals = pool_year[car_perf_cols].median(numeric_only=True)

    if donor_vals is not None and car_perf_cols:
        for c in car_perf_cols:
            if c in donor_vals and pd.notna(donor_vals[c]):
                out.loc[out["driverId"] == driver_id, c] = float(donor_vals[c])

    # recompute driver-vs-team
    if ("driver_avg_finish_last5" in out.columns) and ("team_avg_finish_last5_teammates" in out.columns):
        drv_finish = out.loc[out["driverId"] == driver_id, "driver_avg_finish_last5"].iloc[0]
        team_finish = out.loc[out["driverId"] == driver_id, "team_avg_finish_last5_teammates"].iloc[0]
        if "driver_vs_team_finish_delta_last5" in out.columns:
            out.loc[out["driverId"] == driver_id, "driver_vs_team_finish_delta_last5"] = float(team_finish - drv_finish)
        if "driver_vs_team_finish_ratio_last5" in out.columns:
            out.loc[out["driverId"] == driver_id, "driver_vs_team_finish_ratio_last5"] = safe_div(drv_finish, team_finish, default=1.0)

    if ("driver_avg_points_last5" in out.columns) and ("team_avg_points_last5_teammates" in out.columns):
        drv_pts = out.loc[out["driverId"] == driver_id, "driver_avg_points_last5"].iloc[0]
        team_pts = out.loc[out["driverId"] == driver_id, "team_avg_points_last5_teammates"].iloc[0]
        if "driver_vs_team_points_delta_last5" in out.columns:
            out.loc[out["driverId"] == driver_id, "driver_vs_team_points_delta_last5"] = float(drv_pts - team_pts)
        if "driver_vs_team_points_ratio_last5" in out.columns:
            out.loc[out["driverId"] == driver_id, "driver_vs_team_points_ratio_last5"] = safe_div(drv_pts, team_pts, default=1.0)

    return out


POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# --------------------------------------------------
# REALISTIC SEASON MODE (UNCHANGED)
# --------------------------------------------------
if mode.startswith("Realistic Season"):
    st.subheader("📊 Realistic Season Prediction (engineered rows, 2022–2024 only)")

    if st.button("Predict Season"):
        season_rows = df_app[df_app["year"] == year].copy()
        if season_rows.empty:
            st.warning("No engineered rows found for this season.")
        else:
            race_cols = ["raceId", "round"]
            if "race_name" in season_rows.columns:
                race_cols.append("race_name")

            race_order = (
                season_rows[race_cols]
                .drop_duplicates()
                .sort_values(["round", "raceId"])
            )

            all_predictions = []
            for _, rr in race_order.iterrows():
                race_id = int(rr["raceId"])
                race_name = rr["race_name"] if ("race_name" in rr.index and pd.notna(rr.get("race_name"))) else RACE_ID_TO_NAME.get(race_id, f"raceId={race_id}")

                race_df = season_rows[season_rows["raceId"] == race_id].copy()
                if race_df.empty:
                    continue

                X = make_X(race_df)
                race_df["pred_score"] = MODEL.predict(X)
                race_df = race_df.sort_values("pred_score").reset_index(drop=True)
                race_df["pred_position"] = np.arange(1, len(race_df) + 1)
                race_df["race"] = race_name
                all_predictions.append(race_df)

            if not all_predictions:
                st.warning("No races found to predict for this season.")
                st.stop()

            season_pred = pd.concat(all_predictions, ignore_index=True)

            driver_view = season_pred[season_pred["driverId"] == driver_id].copy()
            st.markdown(f"### {driver_name} — predicted finishes in {year}")
            st.dataframe(driver_view[["race", "pred_position"]], use_container_width=True)

            season_pred["points_calc"] = season_pred["pred_position"].map(POINTS).fillna(0)
            standings = season_pred.groupby("driverId")["points_calc"].sum().sort_values(ascending=False)

            champion_id = int(standings.index[0])
            champion_name = DRIVER_ID_TO_NAME.get(champion_id, f"driverId={champion_id}")
            champ_pts = float(standings.iloc[0])

            st.success(f"🏆 Predicted Champion: {champion_name} ({champ_pts:.0f} pts)")

            top10 = standings.head(10).reset_index()
            top10["driver"] = top10["driverId"].map(lambda x: DRIVER_ID_TO_NAME.get(int(x), f"driverId={x}"))
            top10 = top10[["driver", "points_calc"]].rename(columns={"points_calc": "points"})
            st.markdown("### Top 10 predicted standings")
            st.dataframe(top10, use_container_width=True)

# --------------------------------------------------
# UNSEEN / WHAT-IF MODE (CLEAN UI OUTPUT)
# --------------------------------------------------
else:
    st.subheader("🧪 Unseen / What-If (single race)")

    season_rows = df_app[df_app["year"] == year].copy()
    if season_rows.empty:
        st.warning("No engineered rows found for this season.")
        st.stop()

    # Calendar options (round -> label including race name)
    cal_year = (
        season_rows[["round", "raceId", "race_name"]]
        .drop_duplicates()
        .sort_values(["round", "raceId"])
    )
    cal_year["round"] = cal_year["round"].astype(int)
    cal_year["label"] = cal_year.apply(lambda r: f"R{int(r['round']):02d} — {r['race_name']}", axis=1)

    round_label = st.sidebar.selectbox("Target round (with race name)", cal_year["label"].tolist())
    target_round = int(round_label.split("—")[0].strip().replace("R", ""))

    cand = season_rows[season_rows["round"] == target_round].copy()
    target_race_id = int(cand.sort_values(["raceId"]).iloc[0]["raceId"])
    race_name = cand["race_name"].dropna().iloc[0] if ("race_name" in cand.columns and cand["race_name"].notna().any()) else RACE_ID_TO_NAME.get(target_race_id, f"raceId={target_race_id}")

    # ✅ restrict constructor choices to those appearing in selected year (2022–2024 only)
    ctor_ids_year = sorted(season_rows["constructorId"].dropna().unique().astype(int).tolist())
    ctor_names_year = sorted([CONSTRUCTOR_ID_TO_NAME.get(cid, f"constructorId={cid}") for cid in ctor_ids_year])

    constructor_name = st.selectbox("What-if Constructor (limited to selected year)", ctor_names_year)
    constructor_id = int(CONSTRUCTOR_NAME_TO_ID.get(
        constructor_name,
        next((cid for cid in ctor_ids_year if CONSTRUCTOR_ID_TO_NAME.get(cid) == constructor_name), ctor_ids_year[0])
    ))

    grid_override_on = st.checkbox("Override grid for selected driver", value=True)
    grid_pos = st.slider("Grid position", 1, 20, 10) if grid_override_on else None

    if st.button("Run What-If Prediction", type="primary"):
        race_df = season_rows[season_rows["raceId"] == target_race_id].copy()
        if (race_df["driverId"] == driver_id).sum() == 0:
            st.error("Selected driver not present in this race.")
            st.stop()

        # WHAT-IF: constructor override + car perf + recompute driver-vs-team
        whatif_df = apply_constructor_override_with_car_perf_and_driver_vs_team(
            race_df=race_df.copy(),
            season_rows=season_rows,
            driver_id=driver_id,
            new_constructor_id=constructor_id,
            round_no=int(target_round),
            debug=False  # ✅ no debug output
        )

        if grid_pos is not None and "grid" in whatif_df.columns:
            whatif_df.loc[whatif_df["driverId"] == driver_id, "grid"] = float(grid_pos)

        X_w = make_X(whatif_df)
        w_tmp = whatif_df.copy()
        w_tmp["pred_score"] = MODEL.predict(X_w)
        w_tmp = w_tmp.sort_values("pred_score").reset_index(drop=True)
        w_tmp["pred_position"] = np.arange(1, len(w_tmp) + 1)

        w_row = w_tmp[w_tmp["driverId"] == driver_id].iloc[0]
        w_pred_pos = int(w_row["pred_position"])

        # ✅ Clean UI output (only key fields)
        st.success(f"Predicted finish: **P{w_pred_pos}**")
        st.write({
            "Year": int(year),
            "Track": str(race_name),
            "Driver": str(driver_name),
            "What-if Team": str(constructor_name),
            "Predicted Finish": f"P{w_pred_pos}",
        })
       

# ==================================================
# F1 CHATBOT (FINAL: STRICT FILTERING, ZERO HALLUCINATION)
# ==================================================
import streamlit as st
import pandas as pd
import re

# -----------------------------
# LOAD DATA
# -----------------------------
results = pd.read_csv("archive/results.csv", na_values=[r"\N"])
races = pd.read_csv("archive/races.csv", na_values=[r"\N"])
drivers = pd.read_csv("archive/drivers.csv", na_values=[r"\N"])
constructors = pd.read_csv("archive/constructors.csv", na_values=[r"\N"])
status = pd.read_csv("archive/status.csv", na_values=[r"\N"])

# -----------------------------
# MERGE DATA
# -----------------------------
df = results.merge(races, on="raceId", how="left")
df = df.merge(drivers, on="driverId", how="left")
df = df.merge(constructors, on="constructorId", how="left")
df = df.merge(status, on="statusId", how="left")

# -----------------------------
# CLEAN + NORMALIZE
# -----------------------------
df['driver_name'] = (df['forename'] + " " + df['surname']).str.lower().str.strip()
df['race_name'] = df['name_x'].str.lower().str.strip()
df['constructor_name'] = df['name_y'].str.lower().str.strip()
df['year'] = df['year'].astype(int)
df['position_order'] = df['positionOrder'].astype(int)

# Normalize race names
df['race_name_norm'] = (
    df['race_name']
    .str.replace("grand prix", "", regex=False)
    .str.replace("gp", "", regex=False)
    .str.replace(r"[^a-z\s]", "", regex=True)
    .str.strip()
)

# -----------------------------
# DRIVER NORMALIZATION
# -----------------------------
driver_map = {
    "max": "max verstappen",
    "verstappen": "max verstappen",
    "max verstappen": "max verstappen",

    "lewis": "lewis hamilton",
    "hamilton": "lewis hamilton",
    "lewis hamilton": "lewis hamilton",
}

def normalize_driver(name):
    return driver_map.get(name.lower().strip(), name.lower().strip())

# -----------------------------
# GP NORMALIZATION
# -----------------------------
def normalize_gp(text):
    text = text.lower()
    text = text.replace("grand prix", "").replace("gp", "")
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# -----------------------------
# STRICT RACE MATCHING (FIXED)
# -----------------------------
def get_race_id(gp_text, year):
    if not gp_text or not year:
        return None

    gp_norm = normalize_gp(gp_text)

    season_df = df[df['year'] == year]

    # 1️⃣ EXACT MATCH
    exact = season_df[season_df['race_name_norm'] == gp_norm]
    if not exact.empty:
        return exact.iloc[0]['raceId']

    # 2️⃣ WORD MATCH (SAFE)
    gp_words = set(gp_norm.split())

    for _, row in season_df.iterrows():
        race_words = set(row['race_name_norm'].split())
        if gp_words.issubset(race_words):
            return row['raceId']

    return None

# -----------------------------
# QUERY FUNCTIONS
# -----------------------------
def query_winner(race_id):
    race_df = df[(df['raceId'] == race_id) & (df['position_order'] == 1)]
    if race_df.empty:
        return "❌ No winner found"

    row = race_df.iloc[0]
    return f"Winner of {row['year']} {row['race_name'].title()}: {row['driver_name'].title()}"


def query_podium(drivers_list, race_id):
    race_df = df[df['raceId'] == race_id]

    podium_df = race_df[race_df['position_order'].isin([1,2,3])].sort_values('position_order')

    podium = [
        f"{r['driver_name'].title()} (P{r['position_order']})"
        for _, r in podium_df.iterrows()
    ]

    results = {}
    for d in drivers_list:
        d_norm = normalize_driver(d)
        row = race_df[race_df['driver_name'] == d_norm]

        if row.empty:
            results[d.title()] = None
        else:
            results[d.title()] = int(row.iloc[0]['position_order'])

    lines = []
    for d, pos in results.items():
        if pos is None:
            lines.append(f"{d}: ❌ Not found")
        elif pos <= 3:
            lines.append(f"{d}: ✅ YES (P{pos})")
        else:
            lines.append(f"{d}: ❌ NO (P{pos})")

    # who finished better
    valid = {k: v for k, v in results.items() if v is not None}
    sorted_drivers = sorted(valid, key=lambda x: valid[x])

    better_lines = []
    for i in range(len(sorted_drivers)-1):
        better_lines.append(f"{sorted_drivers[i]} finished better than {sorted_drivers[i+1]}")

    return (
        "🏁 Podium:\n" + ", ".join(podium) + "\n\n"
        + "\n".join(lines)
        + ("\n" + "\n".join(better_lines) if better_lines else "")
    )


def query_position(driver, race_id):
    driver_norm = normalize_driver(driver)
    race_df = df[(df['raceId'] == race_id) & (df['driver_name'] == driver_norm)]

    if race_df.empty:
        return f"{driver.title()}: ❌ Not found"

    pos = int(race_df.iloc[0]['position_order'])
    race_name = race_df.iloc[0]['race_name'].title()

    return f"{driver.title()} finished P{pos} in {race_name}."


# -----------------------------
# UI
# -----------------------------
st.header("💬 F1 Chatbot (Final Accurate Version)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about F1...")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    response = None

    # YEAR
    year_match = re.search(r"\d{4}", user_input)
    year = int(year_match.group()) if year_match else None

    # GP TEXT (FIXED)
    gp_match = re.search(r"([A-Za-z\s]+Grand Prix|[A-Za-z\s]+GP)", user_input, re.IGNORECASE)
    gp_text = gp_match.group(1) if gp_match else ""

    race_id = get_race_id(gp_text, year) if year else None

    # -------------------------
    # INTENT DETECTION (ROBUST)
    # -------------------------
    text = user_input.lower()

    # WINNER
    if race_id and any(x in text for x in ["winner", "who won", "won the race", "p1", "first place"]):
        response = query_winner(race_id)

    # PODIUM
    elif race_id and "podium" in text:
        drivers = re.split(r",|and", text)
        drivers = [d.strip() for d in drivers if len(d.strip()) < 20][:3]
        response = query_podium(drivers, race_id)

    # POSITION
    elif race_id and "position" in text:
        match = re.search(r"position did (.+?) finish", text)
        if match:
            driver = match.group(1)
            response = query_position(driver, race_id)

    # -------------------------
    # FAIL SAFE
    # -------------------------
    if not response:
        response = "❌ Could not match query precisely."

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})