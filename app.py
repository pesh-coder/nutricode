"""
NutriCode Uganda — Streamlit Prototype
Future Makers Hackathon 2026, National Science Week
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

from model import (
    load_training_data, train_models, predict_for_inputs,
    rank_candidate_strains, get_shap_explanation, compute_mineral_ratios,
    FEATURES, FEATURE_LABELS, STRAIN_NAMES, STRAIN_DESCRIPTIONS, CROP_NAMES,
)


# ============================================================
# PAGE CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="NutriCode Uganda",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette matching the pitch deck
FOREST = "#1F4332"
FOREST_DEEP = "#0F2E1F"
AMBER = "#D4932A"
CREAM = "#FAF7F0"
CHARCOAL = "#1E1E1E"
SLATE = "#475569"
RED_ALERT = "#B23A48"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {CREAM};
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    h1, h2, h3 {{
        color: {FOREST_DEEP};
        font-family: Georgia, serif;
    }}
    .stat-card {{
        background: white;
        padding: 1.25rem;
        border-top: 4px solid {AMBER};
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }}
    .stat-number {{
        font-size: 2.4rem;
        font-weight: bold;
        color: {FOREST};
        font-family: Georgia, serif;
        line-height: 1;
    }}
    .stat-label {{
        font-size: 0.85rem;
        color: {SLATE};
        margin-top: 0.5rem;
    }}
    .honest-note {{
        background: #FFF7E1;
        border-left: 4px solid {AMBER};
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        font-style: italic;
        color: {FOREST_DEEP};
        font-size: 0.9rem;
    }}
    .section-tag {{
        color: {AMBER};
        font-weight: bold;
        font-size: 0.75rem;
        letter-spacing: 3px;
        margin-bottom: 0.3rem;
    }}
    .recommendation-card {{
        background: white;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid {FOREST};
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .top-pick {{
        border-left-color: {AMBER} !important;
        background: linear-gradient(to right, #FFF7E1 0%, white 30%);
    }}
    .metric-good {{ color: {FOREST}; font-weight: bold; }}
    .metric-warn {{ color: {AMBER}; font-weight: bold; }}
    .metric-bad  {{ color: {RED_ALERT}; font-weight: bold; }}
    [data-testid="stSidebar"] {{
        background-color: {FOREST_DEEP};
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {AMBER} !important;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 2px;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED DATA & MODELS
# ============================================================

@st.cache_data
def load_districts():
    return pd.read_csv("data/uganda_districts.csv")


@st.cache_resource
def get_trained_models():
    df = load_training_data("data/phytase_performance.csv")
    rf, xgb_model, metrics, X_train = train_models(df)
    return rf, xgb_model, metrics, X_train, df


districts_df = load_districts()
rf_model, xgb_model, metrics, X_train, training_df = get_trained_models()


# ============================================================
# HEADER
# ============================================================

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        f'<div class="section-tag">FUTURE MAKERS HACKATHON 2026  ·  NATIONAL SCIENCE WEEK</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<h1 style="font-size: 2.8rem; margin-top: 0.2rem; margin-bottom: 0;">'
        f'NutriCode <span style="color:{AMBER}; font-style: italic;">Uganda</span></h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="color:{SLATE}; font-size: 1.05rem; margin-top: 0.3rem;">'
        f'AI-powered nutritional intelligence for regionally-adapted phytate reduction in Uganda\'s staple foods.</p>',
        unsafe_allow_html=True
    )
with col_h2:
    st.markdown(
        f'<div style="text-align:right; margin-top: 1.8rem;">'
        f'<div style="color:{SLATE}; font-size: 0.8rem;">Prototype · v0.1</div>'
        f'<div style="color:{FOREST}; font-weight:bold;">Built on {len(training_df)} phytase research samples</div>'
        f'<div style="color:{SLATE}; font-size: 0.8rem;">Ensemble R² = {metrics["ensemble_r2"]:.2f} · MAE = {metrics["ensemble_mae"]:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# ============================================================
# SIDEBAR — INPUTS
# ============================================================

st.sidebar.markdown(
    f'<h2 style="color:{AMBER} !important; font-family:Georgia,serif; margin-top:0;">'
    f'Region Inputs</h2>',
    unsafe_allow_html=True
)

district_name = st.sidebar.selectbox(
    "Target District",
    options=sorted(districts_df["district"].tolist()),
    index=sorted(districts_df["district"].tolist()).index("Gulu"),
    help="Select a Ugandan district for region-specific recommendations."
)

district_row = districts_df[districts_df["district"] == district_name].iloc[0]

st.sidebar.markdown(
    f'<div style="background:{FOREST}; padding: 0.8rem; margin-top: -0.5rem;">'
    f'<div style="color:{AMBER}; font-size:0.7rem; letter-spacing:2px;">DISTRICT PROFILE</div>'
    f'<div style="color:white; font-size:0.85rem; margin-top:0.3rem;">'
    f'Sub-region: <b>{district_row["sub_region"]}</b><br>'
    f'Primary staple: <b>{district_row["primary_staple"]}</b><br>'
    f'Soil: <b>{district_row["soil_type"]}</b> ({district_row["soil_zinc_ppm"]} ppm Zn)<br>'
    f'Child anemia (est.): <b>{district_row["anemia_pct_est"]}%</b><br>'
    f'GAM: <b>{district_row["gam_pct"]}%</b>'
    f'</div></div>',
    unsafe_allow_html=True
)

crop_override = st.sidebar.selectbox(
    "Staple Crop",
    options=list(CROP_NAMES.values()),
    index=list(CROP_NAMES.keys()).index(district_row["primary_staple_code"]),
    help="Override the district's primary staple if modelling a different food."
)
crop_code = [k for k, v in CROP_NAMES.items() if v == crop_override][0]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<div class="section-tag" style="color:{AMBER} !important;">COOKING CONDITIONS</div>',
    unsafe_allow_html=True
)

initial_phytate = st.sidebar.slider(
    "Initial Phytate (mg/100g)",
    min_value=400, max_value=1400, value=850, step=25,
    help="Baseline phytate level in the grain variety. Pearl millet typically 700-900, maize 1000-1200, cassava 500-650."
)

ph_level = st.sidebar.slider(
    "Cooking pH",
    min_value=4.5, max_value=7.0, value=5.4, step=0.1,
    help="pH of the food matrix. Most phytases are most active at pH 5.0-5.5."
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=25, max_value=60, value=37, step=1,
    help="Temperature during enzyme action. Many phytases denature above 55°C."
)

st.sidebar.markdown("---")

about_expander = st.sidebar.expander("About this prototype", expanded=False)
with about_expander:
    st.markdown(f"""
**NutriCode Uganda** is an AI decision-support system for regionally-adapted
phytate reduction in Ugandan staple foods.

This prototype is the **AI layer** of a larger research programme. Laboratory
validation (Phase 1) at Makerere University is the next step.

**Training data:** {len(training_df)} samples from published phytase research
(Greiner, Fredrikson, Hellström, Songré-Ouattara, Nout, Deleu, and others),
combined with Uganda-specific district data from the IPC Acute Malnutrition
Snapshot (April 2025 - February 2026) and UBOS.

**Models:** Random Forest (n=200 trees) + XGBoost ensemble, with SHAP
feature importance for explainability.
    """)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Recommender",
    "🗺️  Uganda Map",
    "🔍 Explainability (SHAP)",
    "📚 Data & Method",
])


# ============================================================
# TAB 1 — RECOMMENDER
# ============================================================

with tab1:
    # Build base inputs (strain_code will be overwritten per candidate)
    base_inputs = {
        "grain_type_code": crop_code,
        "strain_code": 0,
        "initial_phytate_mg": float(initial_phytate),
        "microbe_dose_log": 6,
        "ph_level": float(ph_level),
        "temperature_c": float(temperature),
        "contact_time_min": 1440,
        "soil_zinc_ppm": float(district_row["soil_zinc_ppm"]),
    }

    rankings = rank_candidate_strains(rf_model, xgb_model, base_inputs)

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown(
            f'<div class="section-tag">RANKED RECOMMENDATIONS</div>'
            f'<h2 style="margin-top:0;">Top formulations for {district_name}</h2>'
            f'<p style="color:{SLATE};">Ranked by predicted phytate reduction percentage. '
            f'Target: phytate:iron < 1.0 and phytate:zinc < 15.0 for substantially improved '
            f'mineral absorption.</p>',
            unsafe_allow_html=True
        )

        for idx, row in rankings.iterrows():
            top_class = "recommendation-card top-pick" if idx == 0 else "recommendation-card"
            pi_ratio, pz_ratio, final_phy = compute_mineral_ratios(
                initial_phytate, row["predicted_reduction_pct"]
            )

            # Classify ratios
            pi_class = "metric-good" if pi_ratio < 1.0 else ("metric-warn" if pi_ratio < 3.0 else "metric-bad")
            pz_class = "metric-good" if pz_ratio < 15.0 else ("metric-warn" if pz_ratio < 25.0 else "metric-bad")
            rank_badge = "★ TOP PICK" if idx == 0 else f"#{idx+1}"

            st.markdown(f"""
<div class="{top_class}">
  <div style="display:flex; justify-content:space-between; align-items:baseline;">
    <div>
      <span style="color:{AMBER}; font-weight:bold; font-size:0.8rem; letter-spacing:2px;">{rank_badge}</span>
      <div style="font-weight:bold; font-size:1.1rem; color:{CHARCOAL}; margin-top:0.2rem;">{row["strain_name"]}</div>
    </div>
    <div style="font-size:1.8rem; font-weight:bold; color:{FOREST}; font-family:Georgia,serif;">
      {row["predicted_reduction_pct"]:.1f}%
    </div>
  </div>
  <div style="color:{SLATE}; font-size:0.88rem; margin-top:0.4rem;">{row["strain_description"]}</div>
  <div style="margin-top:0.6rem; font-size:0.85rem;">
    Final phytate:  <b>{final_phy:.0f} mg/100g</b>  &nbsp;·&nbsp;
    Phytate:Iron:  <span class="{pi_class}">{pi_ratio:.2f}</span>  &nbsp;·&nbsp;
    Phytate:Zinc:  <span class="{pz_class}">{pz_ratio:.2f}</span>
  </div>
</div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown(
            f'<div class="section-tag">PROJECTED OUTCOME</div>'
            f'<h2 style="margin-top:0;">Top pick performance</h2>',
            unsafe_allow_html=True
        )
        top = rankings.iloc[0]
        pi_ratio, pz_ratio, final_phy = compute_mineral_ratios(
            initial_phytate, top["predicted_reduction_pct"]
        )
        st.markdown(f"""
<div class="stat-card">
  <div class="stat-number">{top["predicted_reduction_pct"]:.1f}%</div>
  <div class="stat-label">Predicted phytate reduction with {top["strain_name"].split('(')[0].strip()}</div>
</div>
<div class="stat-card">
  <div class="stat-number">{initial_phytate} → {final_phy:.0f}</div>
  <div class="stat-label">Phytate mg/100g (before → after)</div>
</div>
<div class="stat-card">
  <div class="stat-number">{pi_ratio:.2f}</div>
  <div class="stat-label">Phytate:Iron ratio  (target &lt; 1.0)</div>
</div>
<div class="stat-card">
  <div class="stat-number">{pz_ratio:.2f}</div>
  <div class="stat-label">Phytate:Zinc ratio  (target &lt; 15.0)</div>
</div>
        """, unsafe_allow_html=True)

        # Ratio bar chart
        ratio_fig = go.Figure()
        ratios = [("Phytate:Iron", pi_ratio, 1.0), ("Phytate:Zinc", pz_ratio, 15.0)]
        for (label, actual, target) in ratios:
            color = FOREST if actual < target else AMBER if actual < target * 2 else RED_ALERT
            ratio_fig.add_trace(go.Bar(
                x=[actual], y=[label], orientation='h',
                marker_color=color, name=label, showlegend=False,
                text=[f"{actual:.2f}"], textposition="outside"
            ))
            ratio_fig.add_shape(
                type="line", x0=target, x1=target, y0=-0.5, y1=1.5,
                line=dict(color=FOREST_DEEP, width=2, dash="dash")
            )
        ratio_fig.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Against clinical thresholds (dashed = target)", font=dict(size=12, color=SLATE)),
            paper_bgcolor=CREAM, plot_bgcolor=CREAM,
            xaxis=dict(title="Molar ratio", gridcolor="#E5E5E5"),
        )
        st.plotly_chart(ratio_fig, use_container_width=True)

    st.markdown(f"""
<div class="honest-note">
<b>Honest framing:</b> these predictions are the model's best inference from the literature-derived
training data. Real laboratory validation at Makerere University is the next step — Phase 1 of the
NutriCode research programme. Model outputs refine as real Ugandan lab data replaces published estimates.
</div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 2 — UGANDA MAP
# ============================================================

with tab2:
    st.markdown(
        f'<div class="section-tag">REGIONAL NUTRITIONAL RISK</div>'
        f'<h2 style="margin-top:0;">Where intervention is needed most first</h2>'
        f'<p style="color:{SLATE};">Districts coloured by Global Acute Malnutrition (GAM) prevalence. '
        f'Data: IPC Acute Food Insecurity and Malnutrition Snapshot — Uganda, April 2025 – February 2026.</p>',
        unsafe_allow_html=True
    )

    metric_choice = st.radio(
        "Colour metric:",
        ["GAM %", "Anemia %", "Priority Score"],
        horizontal=True,
        index=0
    )
    col_map = {"GAM %": "gam_pct", "Anemia %": "anemia_pct_est", "Priority Score": "priority_score"}[metric_choice]

    # Plotly scatter_geo
    hover_text = [
        f"<b>{row['district']}</b><br>"
        f"Sub-region: {row['sub_region']}<br>"
        f"Staple: {row['primary_staple']}<br>"
        f"Soil Zn: {row['soil_zinc_ppm']} ppm<br>"
        f"GAM: {row['gam_pct']}%<br>"
        f"Anemia: {row['anemia_pct_est']}%<br>"
        f"IPC Phase: {row['ipc_phase']}"
        for _, row in districts_df.iterrows()
    ]

    fig_map = go.Figure(go.Scattergeo(
        lon=districts_df["lon"],
        lat=districts_df["lat"],
        text=hover_text,
        hoverinfo="text",
        marker=dict(
            size=districts_df["total_population"] ** 0.4 / 10 + 8,
            color=districts_df[col_map],
            colorscale=[[0, "#C8E6C9"], [0.5, AMBER], [1, RED_ALERT]],
            colorbar=dict(title=metric_choice, thickness=12),
            line=dict(width=1, color="white"),
        ),
    ))
    fig_map.update_layout(
        geo=dict(
            scope="africa",
            projection_type="mercator",
            center=dict(lat=1.5, lon=32.5),
            lataxis=dict(range=[-1.5, 4.5]),
            lonaxis=dict(range=[29.5, 35.5]),
            showland=True,
            landcolor="#F0EBDE",
            showcountries=True,
            countrycolor="#D0C9B8",
            showsubunits=True,
            subunitcolor="#D0C9B8",
            bgcolor=CREAM,
        ),
        paper_bgcolor=CREAM,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Top-priority districts table
    st.markdown(
        f'<div class="section-tag" style="margin-top:1.5rem;">TOP PRIORITY DISTRICTS</div>'
        f'<h3 style="margin-top:0;">Highest risk — recommended pilot targets</h3>',
        unsafe_allow_html=True
    )
    top_priority = districts_df.nlargest(10, "priority_score")[
        ["district", "sub_region", "primary_staple", "gam_pct", "anemia_pct_est", "ipc_phase", "priority_score"]
    ].rename(columns={
        "district": "District", "sub_region": "Sub-region",
        "primary_staple": "Staple", "gam_pct": "GAM %",
        "anemia_pct_est": "Anemia %", "ipc_phase": "IPC Phase",
        "priority_score": "Priority"
    })
    st.dataframe(top_priority, use_container_width=True, hide_index=True)


# ============================================================
# TAB 3 — SHAP EXPLAINABILITY
# ============================================================

with tab3:
    st.markdown(
        f'<div class="section-tag">EXPLAINABILITY</div>'
        f'<h2 style="margin-top:0;">Why the model recommends what it does</h2>'
        f'<p style="color:{SLATE};">SHAP feature importance reveals exactly how each input — crop, strain, soil, '
        f'cooking conditions — contributes to the predicted phytate reduction. This is critical for scientific '
        f'credibility: a recommendation you cannot explain is a recommendation you cannot defend.</p>',
        unsafe_allow_html=True
    )

    col_s1, col_s2 = st.columns([1, 1])

    # Use the top-ranked strain for the SHAP plot
    top_strain_code = int(rankings.iloc[0]["strain_code"])
    shap_inputs = dict(base_inputs)
    shap_inputs["strain_code"] = top_strain_code
    # Apply defaults for that strain
    if top_strain_code == 0:
        shap_inputs.update({"ph_level": 5.4, "temperature_c": 35, "microbe_dose_log": 6, "contact_time_min": 1440})
    elif top_strain_code == 1:
        shap_inputs.update({"ph_level": 6.0, "temperature_c": 40, "microbe_dose_log": 7, "contact_time_min": 1440})
    elif top_strain_code == 2:
        shap_inputs.update({"ph_level": 5.0, "temperature_c": 37, "microbe_dose_log": 5, "contact_time_min": 1440})
    elif top_strain_code == 3:
        shap_inputs.update({"ph_level": 5.4, "temperature_c": 30, "microbe_dose_log": 6, "contact_time_min": 120})
    else:
        shap_inputs.update({"ph_level": 5.2, "temperature_c": 37, "microbe_dose_log": 6, "contact_time_min": 1440})

    shap_values, base_value, feature_values = get_shap_explanation(rf_model, X_train, shap_inputs)

    with col_s1:
        st.markdown(f'<h3>Feature contributions — top pick</h3>', unsafe_allow_html=True)
        shap_df = pd.DataFrame({
            "feature": [FEATURE_LABELS[f] for f in FEATURES],
            "value": [feature_values[f] for f in FEATURES],
            "shap": shap_values,
        }).sort_values("shap", key=abs, ascending=True)

        colors = [FOREST if v > 0 else RED_ALERT for v in shap_df["shap"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["shap"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in shap_df["shap"]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            height=380,
            margin=dict(l=10, r=40, t=20, b=10),
            xaxis=dict(title="SHAP value (impact on prediction, % points)", gridcolor="#E5E5E5"),
            paper_bgcolor=CREAM, plot_bgcolor=CREAM,
            showlegend=False,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_s2:
        st.markdown(f'<h3>What this means</h3>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:white; padding:1rem; border-left:4px solid {FOREST};">
<p style="color:{CHARCOAL}; font-size:0.92rem;">
<b>Base prediction:</b> {base_value:.1f}% &nbsp; (the average across training samples)<br>
<b>Final prediction:</b> {base_value + shap_values.sum():.1f}%
</p>
<p style="color:{SLATE}; font-size:0.88rem; margin-top:0.8rem;">
Each feature pushes the prediction up (green) or down (red) from the base.
The most important features for this recommendation are shown at the top of the chart.
</p>
</div>
        """, unsafe_allow_html=True)

        # Top 3 drivers
        shap_abs = pd.DataFrame({
            "feature": [FEATURE_LABELS[f] for f in FEATURES],
            "value": [feature_values[f] for f in FEATURES],
            "shap": shap_values,
        }).sort_values("shap", key=abs, ascending=False).head(3)

        st.markdown(f'<div class="section-tag" style="margin-top:1.2rem;">TOP 3 DRIVERS</div>', unsafe_allow_html=True)
        for _, row in shap_abs.iterrows():
            direction = "increases" if row["shap"] > 0 else "decreases"
            arrow = "↑" if row["shap"] > 0 else "↓"
            clr = FOREST if row["shap"] > 0 else RED_ALERT
            st.markdown(f"""
<div style="background:white; padding:0.7rem 1rem; margin-bottom:0.4rem; border-left:3px solid {clr};">
  <div style="font-weight:bold; color:{CHARCOAL};">{arrow} {row["feature"]}</div>
  <div style="font-size:0.85rem; color:{SLATE};">Value: {row["value"]:.2f}  ·  {direction} prediction by {abs(row["shap"]):.2f} pts</div>
</div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
<div class="honest-note">
<b>Why explainability matters:</b> When NutriCode recommends a formulation for Gulu, the Ministry of Health,
Makerere researchers, and district health workers should all be able to see <i>why</i>. A black-box AI
recommendation has no place in a health intervention. SHAP makes the model auditable.
</div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 4 — DATA & METHOD
# ============================================================

with tab4:
    st.markdown(
        f'<div class="section-tag">DATA AND METHOD</div>'
        f'<h2 style="margin-top:0;">How the model was built</h2>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
<div style="background:white; padding:1.2rem; border-top:4px solid {FOREST};">
<h3 style="margin-top:0; color:{FOREST_DEEP};">Training Data</h3>
<p style="color:{CHARCOAL};">
The model is trained on <b>{len(training_df)} samples</b> drawn from published phytase research literature,
covering:
</p>
<ul style="color:{CHARCOAL}; font-size:0.92rem;">
<li>Five microbial strains across lactic acid bacteria, spore-formers, fungi, and yeasts</li>
<li>Five Uganda-relevant staple foods (millet, maize, cassava, sorghum, composite flours)</li>
<li>pH ranges from 4.8 to 6.5, temperatures from 25°C to 45°C</li>
<li>Contact times from 2 hours (yeast rapid action) to 48 hours (traditional fermentation)</li>
<li>Soil zinc profiles from 0.3 to 1.4 ppm reflecting Uganda's regional variability</li>
</ul>
<p style="color:{SLATE}; font-size:0.85rem; margin-top:0.8rem;">
<b>Literature sources:</b> Greiner et al., Fredrikson et al., Hellström et al.,
Songré-Ouattara et al., Nout, Deleu et al.
</p>
</div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
<div style="background:white; padding:1.2rem; border-top:4px solid {AMBER};">
<h3 style="margin-top:0; color:{FOREST_DEEP};">Models & Performance</h3>
<table style="width:100%; color:{CHARCOAL}; font-size:0.95rem;">
<tr><td><b>Random Forest</b></td><td>R² = {metrics["rf_r2"]:.2f}, MAE = {metrics["rf_mae"]:.1f}%</td></tr>
<tr><td><b>XGBoost</b></td><td>R² = {metrics["xgb_r2"]:.2f}, MAE = {metrics["xgb_mae"]:.1f}%</td></tr>
<tr><td><b>Ensemble</b></td><td>R² = {metrics["ensemble_r2"]:.2f}, MAE = {metrics["ensemble_mae"]:.1f}%</td></tr>
<tr><td><b>Train size</b></td><td>{metrics["n_train"]} samples</td></tr>
<tr><td><b>Test size</b></td><td>{metrics["n_test"]} samples (held-out)</td></tr>
</table>
<p style="color:{SLATE}; font-size:0.85rem; margin-top:0.8rem;">
The ensemble averages Random Forest and XGBoost predictions to reduce variance.
SHAP TreeExplainer provides per-prediction feature attribution.
</p>
</div>
        """, unsafe_allow_html=True)

    # Uganda data sources
    st.markdown(f"""
<div style="background:white; padding:1.2rem; border-top:4px solid {FOREST}; margin-top:1rem;">
<h3 style="margin-top:0; color:{FOREST_DEEP};">Uganda Data Sources</h3>
<p style="color:{CHARCOAL};">Regional data integrates {len(districts_df)} Ugandan districts across all 8 major sub-regions:</p>
<ul style="color:{CHARCOAL}; font-size:0.92rem;">
<li><b>IPC Acute Food Insecurity and Malnutrition Snapshot</b> — Uganda, April 2025 – February 2026 (GAM, SAM, phase classifications)</li>
<li><b>Uganda Bureau of Statistics (UBOS)</b> — population figures and administrative structure</li>
<li><b>Uganda Demographic and Health Survey (UDHS)</b> — anemia and stunting baselines</li>
<li><b>UBOS / NIPN Nutrition Data Landscape Report (2020)</b> — regional soil and nutritional linkages</li>
<li><b>Agricultural survey data</b> — staple food by sub-region</li>
</ul>
</div>
    """, unsafe_allow_html=True)

    # Training data preview
    st.markdown(f'<h3 style="margin-top:2rem;">Training data preview</h3>', unsafe_allow_html=True)
    preview = training_df[["strain_name", "crop", "initial_phytate_mg", "ph_level",
                           "temperature_c", "soil_zinc_ppm", "phytate_reduction_pct",
                           "literature_source"]].rename(columns={
        "strain_name": "Strain", "crop": "Crop",
        "initial_phytate_mg": "Phytate (mg/100g)", "ph_level": "pH",
        "temperature_c": "Temp (°C)", "soil_zinc_ppm": "Soil Zn (ppm)",
        "phytate_reduction_pct": "Reduction %", "literature_source": "Source"
    })
    st.dataframe(preview, use_container_width=True, hide_index=True, height=280)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(f"""
<div style="text-align:center; padding: 1rem; color:{SLATE}; font-size:0.88rem;">
<span style="color:{FOREST}; font-family:Georgia,serif; font-style:italic; font-size:1.05rem;">
We are not asking communities to change what they eat. We are using AI to change what they get out of it.
</span>
<br><br>
<b>NutriCode Uganda</b>
</div>
""", unsafe_allow_html=True)
