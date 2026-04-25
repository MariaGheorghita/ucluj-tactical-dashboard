import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances

st.title("Match Analysis")

# =========================
# DATA SOURCE SELECTOR
# =========================

data_mode = st.radio(
    "Select Dataset",
    ["All Matches", "U Cluj Matches"]
)

if data_mode == "All Matches":
    reports = pd.read_csv("all_match_reports.csv")
    vectors = pd.read_csv("all_match_vectors.csv")
else:
    reports = pd.read_csv("ucluj_match_reports.csv")
    vectors = pd.read_csv("ucluj_match_vectors.csv")

# =========================
# FEATURES
# =========================

feature_labels = {
    "progression_index": "Ball Progression",
    "risk_index": "Build-up Risk",
    "final_third_index": "Final Third Presence",
    "defensive_stability_index": "Defensive Stability",
    "pressing_recovery_index": "Pressing Recovery",
    "possession_security_index": "Possession Security",
    "attacking_threat_index": "Attacking Threat"
}

features = list(feature_labels.keys())

cluster_names = {
    0: "High Risk Build-up Match",
    1: "Low Intensity Match",
    2: "Defensive Pressure Match",
    3: "Dominant Attacking Match"
}

# =========================
# SAFETY CHECK
# =========================

required_report_cols = ["match", "cluster"]
required_vector_cols = ["match"] + features

for col in required_report_cols:
    if col not in reports.columns:
        st.error(f"Missing column in reports: {col}")
        st.stop()

for col in required_vector_cols:
    if col not in vectors.columns:
        st.error(f"Missing column in vectors: {col}")
        st.stop()

# =========================
# FILTER U CLUJ
# =========================

if data_mode == "U Cluj Matches":

    reports = reports[
        reports["match"]
        .astype(str)
        .str.contains("Universitatea Cluj", case=False, na=False)
    ].copy()

    vectors = vectors[
        vectors["match"]
        .astype(str)
        .str.contains("Universitatea Cluj", case=False, na=False)
    ].copy()

# =========================
# SYNC DATASETS
# =========================

common_matches = set(reports["match"]).intersection(
    set(vectors["match"])
)

reports = reports[
    reports["match"].isin(common_matches)
].copy()

vectors = vectors[
    vectors["match"].isin(common_matches)
].copy()

if len(reports) == 0 or len(vectors) == 0:
    st.warning("No matching reports and vectors found.")
    st.stop()

reports = reports.reset_index(drop=True)
vectors = vectors.reset_index(drop=True)

# =========================
# MATCH SELECTOR
# =========================

match_list = sorted(
    reports["match"]
    .dropna()
    .unique()
    .tolist()
)

st.caption(f"Matches loaded: {len(match_list)}")

search_text = st.text_input("Search Match")

if search_text:
    match_list = [
        m for m in match_list
        if search_text.lower() in m.lower()
    ]

if len(match_list) == 0:
    st.warning("No match found.")
    st.stop()

selected_match = st.selectbox(
    "Select Match",
    match_list
)

match_row = reports[
    reports["match"] == selected_match
].iloc[0]

vector_row = vectors[
    vectors["match"] == selected_match
].iloc[0]

# =========================
# NORMALIZATION
# =========================

season_avg = vectors[features].mean()

def safe_norm(value, avg):

    if avg == 0:
        return 0

    val = value / avg

    if val > 3:
        val = 3

    if val < 0:
        val = 0

    return val


norm_attack = safe_norm(
    vector_row["attacking_threat_index"],
    season_avg["attacking_threat_index"]
)

norm_progression = safe_norm(
    vector_row["progression_index"],
    season_avg["progression_index"]
)

norm_possession = safe_norm(
    vector_row["possession_security_index"],
    season_avg["possession_security_index"]
)

norm_defense = safe_norm(
    vector_row["defensive_stability_index"],
    season_avg["defensive_stability_index"]
)

norm_pressing = safe_norm(
    vector_row["pressing_recovery_index"],
    season_avg["pressing_recovery_index"]
)

norm_final = safe_norm(
    vector_row["final_third_index"],
    season_avg["final_third_index"]
)

norm_risk = safe_norm(
    vector_row["risk_index"],
    season_avg["risk_index"]
)

# =========================
# SOFASCORE-STYLE SCORE
# =========================

def compute_match_score(
    shooting,
    passing,
    dribbling,
    defending,
    goalkeeping,
    risk
):

    shooting = min(shooting, 2.5)
    passing = min(passing, 2.5)
    dribbling = min(dribbling, 2.5)
    defending = min(defending, 2.5)
    goalkeeping = min(goalkeeping, 2.5)
    risk = min(risk, 2.5)

    raw_rating = (
        0.28 * shooting
        +
        0.20 * passing
        +
        0.18 * dribbling
        +
        0.22 * defending
        +
        0.12 * goalkeeping
        -
        0.10 * risk
    )

    rating = 6.5 + (raw_rating - 1.0) * 2.2

    rating = max(3.0, min(rating, 10.0))

    return round(rating, 2)


team_shooting = (
    norm_attack * 0.65
    +
    norm_final * 0.35
)

team_passing = norm_possession

team_dribbling = norm_progression

team_defending = (
    norm_defense * 0.65
    +
    norm_pressing * 0.35
)

team_goalkeeping = norm_defense

team_score = compute_match_score(
    team_shooting,
    team_passing,
    team_dribbling,
    team_defending,
    team_goalkeeping,
    norm_risk
)

opponent_shooting = safe_norm(
    season_avg["defensive_stability_index"],
    vector_row["defensive_stability_index"]
)

opponent_passing = safe_norm(
    season_avg["possession_security_index"],
    vector_row["possession_security_index"]
)

opponent_dribbling = safe_norm(
    season_avg["pressing_recovery_index"],
    vector_row["pressing_recovery_index"]
)

opponent_defending = safe_norm(
    season_avg["attacking_threat_index"],
    vector_row["attacking_threat_index"]
)

opponent_goalkeeping = opponent_defending

opponent_risk = safe_norm(
    vector_row["risk_index"],
    season_avg["risk_index"]
)

opponent_score = compute_match_score(
    opponent_shooting,
    opponent_passing,
    opponent_dribbling,
    opponent_defending,
    opponent_goalkeeping,
    opponent_risk
)

overall_score = round(
    (team_score + opponent_score) / 2,
    2
)

# =========================
# QUALITY
# =========================

if overall_score >= 8:
    quality = "Excellent Match"
elif overall_score >= 7:
    quality = "Good Match"
elif overall_score >= 6:
    quality = "Average Match"
else:
    quality = "Poor Match"

cluster_id = int(match_row["cluster"])

cluster_label = cluster_names.get(
    cluster_id,
    "Unknown"
)

# =========================
# MAIN MATCH SUMMARY
# =========================

st.header("Match Summary")

st.write(f"**Match:** {selected_match}")
st.write(f"**Cluster Type:** {cluster_label}")
st.write(f"**Match Quality:** {quality}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Team Score",
        f"{team_score} / 10"
    )

with col2:
    st.metric(
        "Opponent Score",
        f"{opponent_score} / 10"
    )

with col3:
    st.metric(
        "Overall Score",
        f"{overall_score} / 10"
    )

st.caption(
    "Scores are inspired by rating systems: shooting, passing, progression, defending, goalkeeping proxy and risk."
)

# =========================
# MAIN TACTICAL CHART
# =========================

st.header("Key Tactical Profile")

main_profile_df = pd.DataFrame({
    "Category": [
        "Shooting",
        "Passing",
        "Progression",
        "Defending",
        "Goalkeeping Proxy",
        "Build-up Risk"
    ],
    "Score": [
        team_shooting,
        team_passing,
        team_dribbling,
        team_defending,
        team_goalkeeping,
        norm_risk
    ]
})

fig_main = px.bar(
    main_profile_df,
    x="Category",
    y="Score",
    title="Main Tactical Categories"
)

st.plotly_chart(
    fig_main,
    use_container_width=True
)

# =========================
# AI SUMMARY
# =========================

st.header("AI Match Summary")

strength_values = {
    "Shooting": team_shooting,
    "Passing": team_passing,
    "Progression": team_dribbling,
    "Defending": team_defending,
    "Pressing Recovery": norm_pressing,
    "Final Third Presence": norm_final
}

weakness_values = {
    "Shooting": team_shooting,
    "Passing": team_passing,
    "Progression": team_dribbling,
    "Defending": team_defending,
    "Pressing Recovery": norm_pressing,
    "Final Third Presence": norm_final,
    "Build-up Risk": 2 - norm_risk
}

main_strength = max(
    strength_values,
    key=strength_values.get
)

main_weakness = min(
    weakness_values,
    key=weakness_values.get
)

summary_text = f"""
This match was classified as **{cluster_label}**.

The strongest tactical area was **{main_strength}**, while the biggest concern was **{main_weakness}**.
The team rating was **{team_score}/10**, compared with an estimated opponent rating of **{opponent_score}/10**.

"""

if norm_risk > 1.15:
    summary_text += """
The main warning sign is the high build-up risk. The team should reduce risky actions in deeper areas and avoid unnecessary losses during progression.
"""

elif team_shooting < 0.9:
    summary_text += """
The main issue is attacking threat. The team reached fewer dangerous situations than expected compared with the season baseline.
"""

elif team_defending < 0.9:
    summary_text += """
The main issue is defensive stability. The team should improve compactness and reduce space between lines.
"""

else:
    summary_text += """
The tactical profile is relatively balanced, with no major structural weakness detected from the available indicators.
"""

st.markdown(summary_text)

# =========================
# TAGS
# =========================

st.subheader("Match Tags")

tags = []

if team_shooting > 1.15:
    tags.append("Strong Shooting")

if team_passing > 1.15:
    tags.append("Secure Passing")

if team_dribbling > 1.15:
    tags.append("Strong Progression")

if team_defending > 1.15:
    tags.append("Strong Defending")

if norm_pressing > 1.15:
    tags.append("Good Pressing Recovery")

if norm_risk > 1.15:
    tags.append("High Build-up Risk")

if len(tags) == 0:
    tags.append("Balanced Match")

st.write(" • ".join(tags))

# =========================
# SIMILAR MATCHES
# =========================

st.header("Similar Matches")

vector_matrix = vectors[features].reset_index(drop=True)

match_index = vectors[
    vectors["match"] == selected_match
].index[0]

distances = euclidean_distances(
    [vector_matrix.iloc[match_index]],
    vector_matrix
)[0]

similar_df = vectors.copy()

similar_df["distance"] = distances

similar_df = similar_df.sort_values(
    by="distance"
)

similar_df = similar_df[
    similar_df["match"] != selected_match
]

top_similar = similar_df.head(3)

if len(top_similar) == 0:
    st.write("No similar matches found.")
else:
    for _, row in top_similar.iterrows():
        st.write("-", row["match"])

# =========================
# ADVANCED DETAILS
# =========================

with st.expander("Advanced tactical charts"):

    st.subheader("Raw Tactical Radar")

    radar_df = pd.DataFrame({
        "Metric": [
            feature_labels[f]
            for f in features
        ],
        "Value": [
            vector_row[f]
            for f in features
        ]
    })

    fig_radar = px.line_polar(
        radar_df,
        r="Value",
        theta="Metric",
        line_close=True,
        title="Raw Tactical Vector"
    )

    fig_radar.update_traces(
        fill="toself"
    )

    st.plotly_chart(
        fig_radar,
        use_container_width=True
    )

    st.subheader("Metric Comparison vs Season")

    comparison_df = pd.DataFrame({
        "Metric": [
            feature_labels[f]
            for f in features
        ],
        "Match Value": [
            vector_row[f]
            for f in features
        ],
        "Season Average": [
            season_avg[f]
            for f in features
        ]
    })

    fig_compare = px.bar(
        comparison_df,
        x="Metric",
        y=[
            "Match Value",
            "Season Average"
        ],
        barmode="group",
        title="Selected Match vs Season Average"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    st.subheader("Detailed Strengths and Weaknesses")

    comparison_values = []

    for f in features:

        diff = vector_row[f] - season_avg[f]

        comparison_values.append({
            "Metric": feature_labels[f],
            "Difference vs Season": diff
        })

    diff_df = pd.DataFrame(comparison_values)

    st.dataframe(
        diff_df.sort_values(
            by="Difference vs Season",
            ascending=False
        ),
        use_container_width=True
    )
