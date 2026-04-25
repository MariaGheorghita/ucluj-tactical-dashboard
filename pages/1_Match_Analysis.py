import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances

# =========================
# PAGE CONFIG + STYLE
# =========================

st.set_page_config(
    page_title="Match Analysis",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.metric-card {
    background: #111827;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid #334155;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #38bdf8;
}

.metric-label {
    color: #94a3b8;
}

.match-card {
    background: #111827;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 15px;
    border: 1px solid #334155;
}

.ai-card {
    background: linear-gradient(135deg, #1d4ed8, #06b6d4);
    padding: 20px;
    border-radius: 18px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================

data_mode = st.sidebar.radio(
    "Dataset",
    ["All Matches", "U Cluj Matches"]
)

if data_mode == "All Matches":
    reports = pd.read_csv("all_match_reports.csv")
    vectors = pd.read_csv("all_match_vectors.csv")
else:
    reports = pd.read_csv("ucluj_match_reports.csv")
    vectors = pd.read_csv("ucluj_match_vectors.csv")

features = [
    "progression_index",
    "risk_index",
    "final_third_index",
    "defensive_stability_index",
    "pressing_recovery_index",
    "possession_security_index",
    "attacking_threat_index"
]

feature_labels = {
    "progression_index": "Progression",
    "risk_index": "Risk",
    "final_third_index": "Final Third",
    "defensive_stability_index": "Defense",
    "pressing_recovery_index": "Pressing",
    "possession_security_index": "Possession",
    "attacking_threat_index": "Attack"
}

# =========================
# MATCH SELECT
# =========================

match_list = sorted(reports["match"].unique())
selected_match = st.sidebar.selectbox("Match", match_list)

match_row = reports[reports["match"] == selected_match].iloc[0]
vector_row = vectors[vectors["match"] == selected_match].iloc[0]

# =========================
# NORMALIZATION
# =========================

season_avg = vectors[features].mean()

def norm(v, avg):
    if avg == 0:
        return 0
    return max(0, min(v / avg, 3))

norm_vals = {f: norm(vector_row[f], season_avg[f]) for f in features}

# =========================
# SCORE MODEL
# =========================

def compute_score(n):

    shooting = n["attacking_threat_index"]*0.6 + n["final_third_index"]*0.4
    passing = n["possession_security_index"]
    progression = n["progression_index"]
    defending = n["defensive_stability_index"]*0.7 + n["pressing_recovery_index"]*0.3
    risk = n["risk_index"]

    raw = (
        0.30 * shooting +
        0.20 * passing +
        0.20 * progression +
        0.25 * defending -
        0.15 * risk
    )

    score = 6.5 + (raw - 1) * 2.5
    return round(max(3, min(score, 10)), 2)

team_score = compute_score(norm_vals)
opponent_score = compute_score({k: 2 - v for k, v in norm_vals.items()})
overall_score = round((team_score + opponent_score) / 2, 2)

# =========================
# HEADER
# =========================

st.markdown(f"""
<div class="match-card">
    <h2>{selected_match}</h2>
    <p>Overall Match Rating: <b>{overall_score}/10</b></p>
</div>
""", unsafe_allow_html=True)

# =========================
# SCORE CARDS
# =========================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{team_score}</div>
        <div class="metric-label">Team Score</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{opponent_score}</div>
        <div class="metric-label">Opponent Score</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{overall_score}</div>
        <div class="metric-label">Match Rating</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# TABS
# =========================

tab1, tab2, tab3 = st.tabs(["Overview", "AI Insight", "Similar"])

# =========================
# OVERVIEW
# =========================

with tab1:

    df = pd.DataFrame({
        "Metric": [feature_labels[f] for f in features],
        "Value": [norm_vals[f] for f in features]
    })

    fig = px.bar(df, x="Metric", y="Value", color="Value")
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

# =========================
# AI INSIGHT
# =========================

with tab2:

    strength = max(norm_vals, key=norm_vals.get)
    weakness = min(norm_vals, key=norm_vals.get)

    st.markdown(f"""
    <div class="ai-card">
        <h3>AI Tactical Insight</h3>
        <p><b>Strongest area:</b> {feature_labels[strength]}</p>
        <p><b>Weakest area:</b> {feature_labels[weakness]}</p>
        <p>Team performed at <b>{team_score}/10</b> level.</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# SIMILAR MATCHES
# =========================

with tab3:

    matrix = vectors[features]
    idx = vectors[vectors["match"] == selected_match].index[0]

    distances = euclidean_distances([matrix.iloc[idx]], matrix)[0]

    vectors["dist"] = distances
    sim = vectors.sort_values("dist")[1:6]

    for _, row in sim.iterrows():
        st.write(row["match"])
