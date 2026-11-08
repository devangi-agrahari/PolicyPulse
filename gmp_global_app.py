# ============================================================
#  PolicyPulse: central bank policy and market reaction research
#  Event studies, statement language (NLP) and scenario analysis
#  for FOMC (US) and Bank of England MPC (UK) decisions.
#  For academic and research purposes only. Not investment advice.
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 1. PAGE SETUP AND STYLE
# ------------------------------------------------------------
st.set_page_config(
    page_title="PolicyPulse | Monetary policy research",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette: slate and navy neutrals, one amber accent, green/red reserved for direction.
BG = "#0E131B"
PANEL = "#151C27"
BORDER = "#252F3F"
TEXT = "#E6E9EF"
MUTED = "#8E98A9"
ACCENT = "#E3A857"
UP = "#3FB27F"
DOWN = "#E0615A"
NEUTRAL = "#7C879A"
SERIES = ["#5B9BD5", "#E3A857", "#7FB77E", "#B07AA1", "#8E98A9", "#D98C6B"]

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="st-"], .stApp {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {BG}; }}
    .block-container {{ padding-top: 4rem; padding-bottom: 2rem; max-width: 1280px; }}
    section[data-testid="stSidebar"] {{
        background: #0B1017;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
    h1, h2, h3, h4 {{ color: {TEXT}; font-weight: 600; letter-spacing: -0.01em; }}
    div[data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {MUTED}; font-size: 0.78rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.04em;
    }}
    div[data-testid="stMetricValue"] {{ color: {TEXT}; font-weight: 600; }}
    button[data-baseweb="tab"] {{ font-weight: 500; }}
    .eyebrow {{
        color: {ACCENT}; font-size: 0.78rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px;
    }}
    .page-title {{ font-size: 2rem; font-weight: 600; color: {TEXT}; margin: 0; line-height: 1.2; }}
    .page-sub {{ color: {MUTED}; font-size: 0.95rem; margin: 4px 0 18px 0; }}
    .side-label {{
        color: {MUTED}; font-size: 0.74rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em; margin: 14px 0 -4px 0;
    }}
    .note {{ color: {MUTED}; font-size: 0.85rem; line-height: 1.5; }}
    #MainMenu, footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 2. LABELS
# ------------------------------------------------------------
NAMES = {
    "FEDFUNDS": "Fed funds rate (%)",
    "DGS10": "10-year Treasury yield (%)",
    "UNRATE": "Unemployment rate (%)",
    "CPIAUCSL": "CPI (index)",
    "M2SL": "M2 money supply ($bn)",
    "PCEPI": "PCE price index",
    "GDPC1": "Real GDP ($bn)",
    "INDPRO": "Industrial production (index)",
    "^GSPC": "S&P 500 (index)",
    "^VIX": "VIX (index)",
    "BoE_Rate_Pre": "BoE rate before decision (%)",
    "BoE_Rate_Post": "BoE rate after decision (%)",
    "UK10Y_Current": "UK 10-year gilt yield (%)",
    "FTSE_Day0": "FTSE 100 announcement-day return (%)",
    "FTSE_CAR": "FTSE 100 cumulative abnormal return (%)",
    "SP500_Day0": "S&P 500 announcement-day return (%)",
    "SP500_CAR": "S&P 500 cumulative abnormal return (%)",
    "Rate_Change_bp": "Rate change (bp)",
    "is_hike": "Rate hike (0/1)",
    "is_cut": "Rate cut (0/1)",
    "sentiment_score": "Positive-tone score (0 to 1)",
    "count_inflation": "Inflation mentions",
    "count_employment": "Employment mentions",
    "count_growth": "Growth mentions",
    "count_risk": "Risk mentions",
    "count_financial": "Financial-conditions mentions",
}


def nice(col):
    return NAMES.get(col, col.replace("_", " "))


SECTORS = {
    "SP500": "S&P 500", "Tech": "Technology", "Financials": "Financials", "Energy": "Energy",
    "Industrials": "Industrials", "Utilities": "Utilities", "HealthCare": "Health care",
    "Discretionary": "Consumer discretionary", "Staples": "Consumer staples",
    "RealEstate": "Real estate", "Materials": "Materials", "Communications": "Communication services",
}


# ------------------------------------------------------------
# 3. DATA AND MODEL
# ------------------------------------------------------------
def safe_read_csv(path):
    """Reads a CSV without crashing; returns an empty frame on failure."""
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_all_data():
    """Loads the four research datasets and guarantees the columns the app needs."""
    data = {
        "us_macro": safe_read_csv("us_monetary_data_phase1.csv"),
        "us_event": safe_read_csv("phase2_event_study_output.csv"),
        "us_nlp": safe_read_csv("phase3_fomc_nlp_features_2010plus.csv"),
        "uk_model": safe_read_csv("uk_phase4_model_data.csv"),
    }

    req_uk = ["Rate_Change_bp", "is_hike", "is_cut", "FTSE_CAR", "FTSE_Day0", "BoE_Rate_Current", "UK10Y_Current"]
    req_us = ["sentiment_score", "count_inflation", "count_employment", "count_risk", "FEDFUNDS", "DGS10", "SP500_Day0"]

    if data["uk_model"].empty or len(data["uk_model"].columns) < 2:
        dates = pd.date_range("2010-01-01", "2023-01-01", freq="QE")
        data["uk_model"] = pd.DataFrame(0.0, index=dates, columns=req_uk)
    else:
        for c in req_uk:
            if c not in data["uk_model"].columns:
                data["uk_model"][c] = 0.0
    for c in req_uk:
        data["uk_model"][c] = pd.to_numeric(data["uk_model"][c], errors="coerce").fillna(0.0)

    if not data["us_event"].empty and not data["us_nlp"].empty:
        combined = data["us_event"].join(data["us_nlp"], how="inner")
        if not data["us_macro"].empty:
            macro_aligned = data["us_macro"].reindex(combined.index, method="ffill")
            data["us_full"] = combined.join(macro_aligned, how="left")
        else:
            data["us_full"] = combined
    else:
        dates = pd.date_range("2010-01-01", "2023-01-01", freq="ME")
        data["us_full"] = pd.DataFrame(0.0, index=dates, columns=req_us)

    for c in req_us:
        if c not in data["us_full"].columns:
            data["us_full"][c] = 0.0

    data["us_full"] = data["us_full"].fillna(0.0)
    data["uk_model"] = data["uk_model"].fillna(0.0)
    return data


DATA = load_all_data()


@st.cache_resource
def get_model(region="USA"):
    """Fits a Random Forest on the event sample (in-sample, for illustration)."""
    if region == "USA":
        df = DATA["us_full"]
        features = ["sentiment_score", "count_inflation", "count_employment", "count_risk", "FEDFUNDS", "DGS10"]
        target = "SP500_Day0"
    else:
        df = DATA["uk_model"]
        features = ["Rate_Change_bp", "is_hike", "is_cut", "FTSE_CAR"]
        target = "FTSE_Day0"

    if df.empty or len(df) < 5:
        return None, None, features

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X, df[target])
    return model, scaler, features


def latest(df, col):
    """Most recent non-missing value of a column, or None."""
    if col in df.columns:
        s = df[col].dropna()
        if len(s):
            return float(s.iloc[-1])
    return None


def sample_span(df):
    try:
        return f"{df.index.min():%b %Y} to {df.index.max():%b %Y}"
    except Exception:
        return "n/a"


# ------------------------------------------------------------
# 4. CHART STYLING
# ------------------------------------------------------------
def style(fig, title=None, height=380, legend=True, x_title=None, y_title=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Inter, sans-serif", size=12),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=TEXT)) if title else None,
        height=height,
        margin=dict(l=4, r=4, t=52 if title else 16, b=4),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="", font=dict(size=12)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT)),
        colorway=SERIES,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, zeroline=False, title=x_title)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, zeroline=False, title=y_title)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def tab_intro(text):
    st.markdown(f"<div class='note' style='margin-bottom:14px'>{text}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 5. SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown("<div class='eyebrow'>PolicyPulse</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:1.15rem;font-weight:600;color:#E6E9EF;margin-bottom:6px'>Research terminal</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='side-label'>Central bank</div>", unsafe_allow_html=True)
region_mode = st.sidebar.radio(
    "Central bank",
    ["United States (FOMC)", "United Kingdom (BoE MPC)"],
    label_visibility="collapsed",
)
is_us = region_mode.startswith("United States")
region_key = "USA" if is_us else "UK"

st.sidebar.markdown("<div class='side-label'>Scenario inputs</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='note'>These drive the Scenario analysis tab.</div>", unsafe_allow_html=True)

if is_us:
    s_crash = st.sidebar.slider("Market stress", 0, 10, 2,
                                help="Higher stress lowers the tone of the statement and raises risk language.")
    s_inf = st.sidebar.slider("Inflation language", 0, 25, 5, help="Number of inflation mentions in the statement.")
    s_emp = st.sidebar.slider("Employment language", 0, 25, 5, help="Number of employment mentions in the statement.")
    s_rate = st.sidebar.slider("Fed funds rate (%)", 0.0, 12.0, 5.25, step=0.25)
    s_yield = st.sidebar.slider("10-year yield (%)", 0.0, 12.0, 4.50, step=0.05)

    base_sent = 0.5 - (s_crash * 0.15)
    base_risk = 5 + (s_crash * 3)
    overrides = {
        "sentiment_score": max(0.0, min(1.0, base_sent)),
        "count_risk": base_risk,
        "count_inflation": s_inf,
        "count_employment": s_emp,
        "FEDFUNDS": s_rate,
        "DGS10": s_yield,
    }
else:
    s_bp = st.sidebar.slider("Rate change (bp)", -100, 100, 0, step=25)
    s_car = st.sidebar.slider("FTSE 100 trend (CAR %)", -5.0, 5.0, 0.0, step=0.25)
    overrides = {
        "Rate_Change_bp": s_bp,
        "is_hike": 1 if s_bp > 0 else 0,
        "is_cut": 1 if s_bp < 0 else 0,
        "FTSE_CAR": s_car,
    }

st.sidebar.markdown("<hr style='border-color:#252F3F;margin:22px 0 10px 0'>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='note'>For academic and research purposes only. Not investment advice.</div>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 6. HEADER AND KEY FIGURES
# ------------------------------------------------------------
st.markdown("<div class='eyebrow'>Monetary policy research</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='page-title'>{'Federal Reserve' if is_us else 'Bank of England'}: policy decisions and market reaction</div>",
    unsafe_allow_html=True,
)

if is_us:
    ev = DATA["us_event"]
    st.markdown(
        f"<div class='page-sub'>FOMC announcements, {sample_span(ev)}. Event study, statement language and scenario model.</div>",
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    v = latest(DATA["us_macro"], "FEDFUNDS")
    k1.metric("Fed funds rate", f"{v:.2f}%" if v is not None else "n/a")
    v = latest(DATA["us_macro"], "DGS10")
    k2.metric("10-year yield", f"{v:.2f}%" if v is not None else "n/a")
    k3.metric("FOMC events studied", f"{len(ev)}")
    if "SP500_Day0" in ev.columns and ev["SP500_Day0"].notna().any():
        k4.metric("Avg. S&P 500 move on day", f"{ev['SP500_Day0'].mean():+.2f}%")
    else:
        k4.metric("Avg. S&P 500 move on day", "n/a")
else:
    uk = DATA["uk_model"]
    st.markdown(
        f"<div class='page-sub'>MPC decisions, {sample_span(uk)}. Event study and scenario model.</div>",
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    col = "BoE_Rate_Post" if "BoE_Rate_Post" in uk.columns else "BoE_Rate_Current"
    k1.metric("Last recorded BoE rate", f"{uk[col].iloc[-1]:.2f}%")
    k2.metric("Decisions studied", f"{len(uk)}")
    k3.metric("Hikes / cuts", f"{int(uk['is_hike'].sum())} / {int(uk['is_cut'].sum())}")
    k4.metric("Avg. FTSE 100 move on day", f"{uk['FTSE_Day0'].mean():+.2f}%")

st.write("")

tabs = st.tabs([
    "Overview",
    "Announcement reactions",
    "Statement language",
    "Model drivers",
    "Scenario analysis",
    "3D explorer",
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tabs[0]:
    tab_intro("Policy rates and related indicators over time. Pick one or more series to compare; "
              "series are on different scales, so compare direction rather than level.")
    df_macro = DATA["us_macro"] if is_us else DATA["uk_model"]
    num = df_macro.select_dtypes(include=np.number)
    cols = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
    if not is_us:
        preferred = [c for c in ["BoE_Rate_Post", "BoE_Rate_Pre", "UK10Y_Current", "FTSE_CAR"] if c in cols]
        cols = preferred + [c for c in cols if c not in preferred]
    if cols:
        default = [c for c in (["FEDFUNDS", "DGS10"] if is_us else ["BoE_Rate_Pre", "BoE_Rate_Post"]) if c in cols] or cols[:2]
        selected = st.multiselect("Series", cols, default=default, format_func=nice)
        if selected:
            plot_df = df_macro[selected].rename(columns=nice)
            fig = px.line(plot_df, color_discrete_sequence=SERIES)
            fig.update_traces(connectgaps=True, line=dict(width=2))
            fig.update_layout(hovermode="x unified")
            show(style(fig, height=420, x_title=None, y_title=None))
    else:
        st.info("No numeric data available for this market.")


# ------------------------------------------------------------
# TAB 2: ANNOUNCEMENT REACTIONS
# ------------------------------------------------------------
with tabs[1]:
    if is_us:
        tab_intro("How equities moved on FOMC announcement days. Returns are in percent; "
                  "the sector chart shows which parts of the market react most.")
        df_ev = DATA["us_event"]
        metric = "SP500_Day0" if "SP500_Day0" in df_ev.columns else (df_ev.columns[0] if len(df_ev.columns) else None)
    else:
        tab_intro("How the FTSE 100 moved on MPC decision days, and whether the direction of the "
                  "rate decision made a difference. Returns are in percent.")
        df_ev = DATA["uk_model"]
        metric = "FTSE_Day0"

    if metric and not df_ev.empty and metric in df_ev.columns:
        s = df_ev[metric].dropna()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average move", f"{s.mean():+.2f}%")
        m2.metric("Median move", f"{s.median():+.2f}%")
        m3.metric("Volatility (std. dev.)", f"{s.std():.2f}%")
        m4.metric("Share of up days", f"{(s > 0).mean() * 100:.0f}%")
        st.write("")

        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(df_ev, x=metric, nbins=30, color_discrete_sequence=[SERIES[0]])
            fig_hist.add_vline(x=float(s.mean()), line_dash="dash", line_color=ACCENT,
                               annotation_text="mean", annotation_font_color=ACCENT)
            fig_hist.update_traces(marker_line_width=0)
            show(style(fig_hist, title="Distribution of announcement-day returns", legend=False,
                       x_title="Return (%)", y_title="Number of events"))
        with c2:
            if is_us:
                sec_cols = [c for c in df_ev.columns if c.endswith("_Day0") and c.replace("_Day0", "") in SECTORS]
                avg = df_ev[sec_cols].mean().rename(lambda c: SECTORS[c.replace("_Day0", "")]).sort_values()
                fig_sec = go.Figure(go.Bar(
                    x=avg.values, y=avg.index, orientation="h",
                    marker_color=[UP if v >= 0 else DOWN for v in avg.values],
                    hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
                ))
                show(style(fig_sec, title="Average announcement-day move by sector", legend=False,
                           x_title="Average return (%)"))
            else:
                kind = np.where(df_ev["is_hike"] == 1, "Hike", np.where(df_ev["is_cut"] == 1, "Cut", "Hold"))
                plot_df = pd.DataFrame({"Decision": kind, "Return (%)": df_ev[metric].values})
                fig_box = px.box(plot_df, x="Decision", y="Return (%)", color="Decision", points="all",
                                 category_orders={"Decision": [d for d in ["Cut", "Hold", "Hike"] if d in set(kind)]},
                                 color_discrete_map={"Hike": DOWN, "Cut": UP, "Hold": NEUTRAL})
                show(style(fig_box, title="FTSE 100 return by decision type", legend=False, x_title=None))
    else:
        st.info("No event data available for this market.")


# ------------------------------------------------------------
# TAB 3: STATEMENT LANGUAGE
# ------------------------------------------------------------
with tabs[2]:
    if is_us:
        tab_intro("What FOMC statements talk about and how positive they sound, scored with a "
                  "finance-tuned language model. Most statements are neutral in tone.")
        df_nlp = DATA["us_nlp"]
        if not df_nlp.empty and "sentiment_score" in df_nlp.columns:
            tone_lbl = df_nlp["label"] if "label" in df_nlp.columns else pd.Series(
                np.where(df_nlp["sentiment_score"] > 0, "positive", "neutral"), index=df_nlp.index)
            tone_df = pd.DataFrame({"Score": df_nlp["sentiment_score"], "Tone": tone_lbl.astype(str).str.capitalize()})
            n_pos = int((tone_df["Tone"] == "Positive").sum())
            fig_tone = px.scatter(tone_df, y="Score", color="Tone",
                                  color_discrete_map={"Positive": UP, "Neutral": NEUTRAL, "Negative": DOWN})
            fig_tone.update_traces(marker=dict(size=9, line=dict(width=0)))
            show(style(fig_tone, title=f"Tone of each FOMC statement ({n_pos} of {len(tone_df)} scored positive)",
                       height=300, y_title="Positive-tone score (0 to 1)"))

            topics = [c for c in ["count_inflation", "count_employment", "count_growth", "count_risk"] if c in df_nlp.columns]
            if topics:
                smooth = df_nlp[topics].rolling(4, min_periods=1).mean().rename(columns=nice)
                # Break the line across any long stretch with no statements in the data
                idx = smooth.index
                gap_note = ""
                for i in range(1, len(idx)):
                    if (idx[i] - idx[i - 1]).days > 365:
                        smooth.loc[idx[i] - pd.Timedelta(days=30)] = np.nan
                        gap_note = f"No statement data in this file between {idx[i - 1]:%b %Y} and {idx[i]:%b %Y}."
                smooth = smooth.sort_index()
                fig_topic = px.line(smooth, color_discrete_sequence=SERIES)
                fig_topic.update_traces(line=dict(width=2))
                fig_topic.update_layout(hovermode="x unified")
                show(style(fig_topic, title="Topic emphasis (4-meeting rolling average of keyword counts)",
                           height=340, y_title="Mentions per statement"))
                if gap_note:
                    st.caption(gap_note.strip())

            both = DATA["us_full"][["sentiment_score", "SP500_Day0"]]
            both = both[(both["sentiment_score"] != 0) | (both["SP500_Day0"] != 0)]
            if len(both) > 5:
                corr = both["sentiment_score"].corr(both["SP500_Day0"])
                fig_sc = px.scatter(both, x="sentiment_score", y="SP500_Day0", opacity=0.75,
                                    color_discrete_sequence=[SERIES[1]])
                slope, intercept = np.polyfit(both["sentiment_score"], both["SP500_Day0"], 1)
                xs = np.array([both["sentiment_score"].min(), both["sentiment_score"].max()])
                fig_sc.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                            line=dict(color=MUTED, dash="dash"), name="Linear fit"))
                show(style(fig_sc, title=f"Does tone matter? Correlation with S&P 500 move: {corr:+.2f}",
                           legend=False, height=340, x_title="Positive-tone score",
                           y_title="S&P 500 announcement-day return (%)"))
        else:
            st.info("No statement data available.")
    else:
        st.info("Statement language analysis is available for the United States only. "
                "Switch to United States (FOMC) in the sidebar.")


# ------------------------------------------------------------
# TAB 4: MODEL DRIVERS
# ------------------------------------------------------------
with tabs[3]:
    tab_intro("Which inputs the Random Forest relies on most when estimating the announcement-day return. "
              "Importance is relative and describes the fitted model, not cause and effect. "
              "The model is fit on the full event sample and is illustrative, not a forecast.")
    model, scaler, feats = get_model(region_key)
    if model:
        imp = pd.DataFrame({"Feature": [nice(f) for f in feats], "Importance": model.feature_importances_})
        imp = imp.sort_values("Importance")
        fig_imp = go.Figure(go.Bar(
            x=imp["Importance"], y=imp["Feature"], orientation="h", marker_color=SERIES[0],
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig_imp.update_xaxes(tickformat=".0%")
        show(style(fig_imp, title="Relative importance of model inputs", legend=False, x_title="Share of importance"))
    else:
        st.info("Not enough data to fit the model.")


# ------------------------------------------------------------
# TAB 5: SCENARIO ANALYSIS
# ------------------------------------------------------------
with tabs[4]:
    tab_intro("Set the inputs in the sidebar to see the announcement-day return the model implies, "
              "and where it sits relative to history.")
    model, scaler, feats = get_model(region_key)
    if model:
        base_df = DATA["us_full"] if is_us else DATA["uk_model"]
        target = "SP500_Day0" if is_us else "FTSE_Day0"
        base = base_df.iloc[-1].copy()
        latest_vals = base[feats].astype(float).copy()
        for k, v in overrides.items():
            base[k] = v

        X_in = pd.DataFrame([base[feats].values.astype(float)], columns=feats)
        pred = float(model.predict(scaler.transform(X_in))[0])
        hist = base_df[target]
        avg_hist = float(hist.mean())
        pct = float((hist < pred).mean() * 100)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model-implied return", f"{pred:+.2f}%")
        c2.metric("Versus historical average", f"{pred - avg_hist:+.2f} pp")
        c3.metric("Percentile in history", f"{pct:.0f}th")
        st.write("")

        left, right = st.columns([3, 2])
        with left:
            fig_h = px.histogram(hist, nbins=30, color_discrete_sequence=[SERIES[0]])
            fig_h.update_traces(marker_line_width=0)
            fig_h.add_vline(x=pred, line_color=UP if pred >= 0 else DOWN, line_width=2,
                            annotation_text="scenario", annotation_position="top left",
                            annotation_font_color=UP if pred >= 0 else DOWN)
            fig_h.add_vline(x=avg_hist, line_dash="dash", line_color=MUTED,
                            annotation_text="average", annotation_position="top right",
                            annotation_font_color=MUTED)
            show(style(fig_h, title="Scenario versus historical announcement-day returns", legend=False,
                       x_title="Return (%)", y_title="Number of events"))
        with right:
            compare = pd.DataFrame({
                "Input": [nice(f) for f in feats],
                "Latest in data": [round(float(latest_vals[f]), 2) for f in feats],
                "Scenario": [round(float(base[f]), 2) for f in feats],
            })
            st.markdown("<div style='color:#E6E9EF;font-weight:600;font-size:0.95rem;margin:6px 0 8px 0'>Inputs used</div>",
                        unsafe_allow_html=True)
            st.dataframe(compare, hide_index=True, use_container_width=True)
    else:
        st.info("Not enough data to fit the model.")


# ------------------------------------------------------------
# TAB 6: 3D EXPLORER
# ------------------------------------------------------------
with tabs[5]:
    tab_intro("Explore how any three variables relate. Drag to rotate, scroll to zoom.")
    if is_us:
        df_3d = DATA["us_full"]
        dx, dy, dz = "FEDFUNDS", "sentiment_score", "SP500_Day0"
    else:
        df_3d = DATA["uk_model"]
        dx, dy, dz = "Rate_Change_bp", "FTSE_CAR", "FTSE_Day0"

    num3 = df_3d.select_dtypes(include=np.number)
    cols_3d = [c for c in num3.columns if num3[c].nunique(dropna=True) > 1]
    if len(cols_3d) >= 3:
        c1, c2, c3 = st.columns(3)
        x_ax = c1.selectbox("X axis", cols_3d, index=cols_3d.index(dx) if dx in cols_3d else 0, format_func=nice)
        y_ax = c2.selectbox("Y axis", cols_3d, index=cols_3d.index(dy) if dy in cols_3d else 1, format_func=nice)
        z_ax = c3.selectbox("Z axis", cols_3d, index=cols_3d.index(dz) if dz in cols_3d else 2, format_func=nice)

        fig_3d = px.scatter_3d(df_3d, x=x_ax, y=y_ax, z=z_ax, color=z_ax, opacity=0.8,
                               color_continuous_scale="Viridis",
                               labels={x_ax: nice(x_ax), y_ax: nice(y_ax), z_ax: nice(z_ax)})
        fig_3d.update_traces(marker=dict(size=4))
        axis_style = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)",
                          zerolinecolor="rgba(255,255,255,0.12)", color=MUTED)
        fig_3d.update_layout(scene=dict(xaxis=axis_style, yaxis=axis_style, zaxis=axis_style, bgcolor="rgba(0,0,0,0)"),
                             coloraxis_showscale=False)
        show(style(fig_3d, height=620, legend=False))
    else:
        st.info("Not enough variables to plot.")


st.markdown("<hr style='border-color:#252F3F;margin-top:2rem'>", unsafe_allow_html=True)
st.markdown(
    "<div class='note'>Data: Federal Reserve Economic Data, FOMC statements (2010 onward), Bank of England policy rates, "
    "equity and index prices from Yahoo Finance. Built by Devangi Agrahari and Bhanu Teja Dunna.</div>",
    unsafe_allow_html=True,
)
