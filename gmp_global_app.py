# ============================================================
#  PolicyPulse
#  How do markets react when central banks speak?
#  An event study of Federal Reserve (FOMC) and Bank of England (MPC)
#  decisions: market data, statement language (NLP) and machine learning.
#  For academic and research purposes only. Not investment advice.
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

REPO_URL = "https://github.com/devangi-agrahari/PolicyPulse"

# ------------------------------------------------------------
# 1. PAGE SETUP AND DESIGN SYSTEM
# ------------------------------------------------------------
st.set_page_config(
    page_title="PolicyPulse | Central bank policy and market reaction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BG = "#0D1219"
PANEL = "#141B26"
PANEL_HI = "#192232"
BORDER = "#222C3B"
TEXT = "#E8EBF1"
MUTED = "#8C97A9"
FAINT = "#5E6A7E"
ACCENT = "#E3A857"
BLUE = "#5B9BD5"
UP = "#3FB27F"
DOWN = "#E0615A"
NEUTRAL = "#7C879A"
SERIES = [BLUE, ACCENT, "#7FB77E", "#B07AA1", "#8E98A9", "#D98C6B"]

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]):not(code):not(pre) {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {BG}; }}
    .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }}

    /* hide Streamlit chrome so the product speaks for itself */
    header[data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
    section[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}

    h1, h2, h3, h4 {{ color: {TEXT}; font-weight: 600; letter-spacing: -0.01em; }}

    /* hero */
    .eyebrow {{ color: {ACCENT}; font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }}
    .hero-title {{ font-size: 2.9rem; font-weight: 700; color: {TEXT}; letter-spacing: -0.03em; line-height: 1.05; margin: 6px 0 10px 0; }}
    .hero-sub {{ color: {MUTED}; font-size: 1.06rem; line-height: 1.55; max-width: 780px; margin-bottom: 14px; }}
    .chip {{
        display: inline-block; padding: 4px 11px; margin: 0 6px 6px 0; border-radius: 999px;
        border: 1px solid {BORDER}; background: {PANEL}; color: {MUTED}; font-size: 0.78rem; font-weight: 500;
    }}
    .chip b {{ color: {TEXT}; font-weight: 600; }}

    /* findings cards */
    .fcard {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 18px 18px 16px 18px; height: 100%; min-height: 176px;
    }}
    .fcard .lab {{ color: {MUTED}; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }}
    .fcard .val {{ color: {TEXT}; font-size: 1.65rem; font-weight: 650; letter-spacing: -0.02em; margin: 8px 0 6px 0; line-height: 1.15; }}
    .fcard .sub {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.45; }}

    /* section headings above charts */
    .sec {{ margin: 26px 0 2px 0; color: {TEXT}; font-size: 1.12rem; font-weight: 600; letter-spacing: -0.01em; }}
    .sec-sub {{ color: {MUTED}; font-size: 0.9rem; line-height: 1.5; margin-bottom: 8px; max-width: 820px; }}
    .takeaway {{
        border-left: 3px solid {ACCENT}; background: {PANEL}; padding: 11px 16px; border-radius: 0 8px 8px 0;
        color: {TEXT}; font-size: 0.92rem; line-height: 1.5; margin: 10px 0 4px 0;
    }}
    .takeaway span {{ color: {MUTED}; }}
    .note {{ color: {MUTED}; font-size: 0.84rem; line-height: 1.55; }}

    /* result card in scenario */
    .result {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 22px; }}
    .result .big {{ font-size: 3.1rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; }}
    .result .lab {{ color: {MUTED}; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 10px; }}
    .pbar {{ position: relative; height: 8px; border-radius: 4px; background: {BORDER}; margin: 20px 0 8px 0; }}
    .pdot {{ position: absolute; top: -5px; width: 18px; height: 18px; border-radius: 50%; border: 3px solid {BG}; transform: translateX(-50%); }}
    .pticks {{ display: flex; justify-content: space-between; color: {FAINT}; font-size: 0.72rem; }}

    /* native widgets */
    div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px; }}
    div[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
    div[data-testid="stMetricValue"] {{ color: {TEXT}; font-weight: 650; }}
    button[data-baseweb="tab"] {{ font-weight: 500; padding-left: 14px; padding-right: 14px; }}
    div[data-testid="stTabs"] {{ margin-top: 6px; }}
    div[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 10px; background: {PANEL}; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px; }}
    hr {{ border-color: {BORDER}; }}
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
    "Tech": "Technology", "Financials": "Financials", "Energy": "Energy", "Industrials": "Industrials",
    "Utilities": "Utilities", "HealthCare": "Health care", "Discretionary": "Consumer discretionary",
    "Staples": "Consumer staples", "RealEstate": "Real estate", "Materials": "Materials",
    "Communications": "Communication services",
}


# ------------------------------------------------------------
# 3. DATA AND MODEL
# ------------------------------------------------------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_all_data():
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
    if col in df.columns:
        s = df[col].dropna()
        if len(s):
            return float(s.iloc[-1])
    return None


def span(df):
    try:
        return f"{df.index.min():%b %Y} to {df.index.max():%b %Y}"
    except Exception:
        return "n/a"


def kde(values, points=200):
    """Gaussian kernel density estimate (Silverman bandwidth), no scipy needed."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3 or v.std() == 0:
        return None, None
    h = 1.06 * v.std() * len(v) ** (-1 / 5)
    xs = np.linspace(v.min() - 3 * h, v.max() + 3 * h, points)
    ys = np.exp(-0.5 * ((xs[:, None] - v[None, :]) / h) ** 2).sum(axis=1) / (len(v) * h * np.sqrt(2 * np.pi))
    return xs, ys


# ------------------------------------------------------------
# 4. UI HELPERS
# ------------------------------------------------------------
def style(fig, height=380, legend=True, x_title=None, y_title=None, y_ticks=True):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=4, r=8, t=18, b=4),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="",
                    font=dict(size=12, color=TEXT)),
        hoverlabel=dict(bgcolor=PANEL_HI, bordercolor=BORDER, font=dict(color=TEXT, family="Inter")),
        colorway=SERIES,
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, zeroline=False, title=x_title, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.045)", zeroline=False, showline=False, title=y_title,
                     tickfont=dict(size=11), showticklabels=y_ticks)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def section(title, sub=None):
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='sec-sub'>{sub}</div>", unsafe_allow_html=True)


def takeaway(text, lead="What this shows. "):
    st.markdown(f"<div class='takeaway'><span>{lead}</span>{text}</div>", unsafe_allow_html=True)


def card(label, value, sub):
    return f"<div class='fcard'><div class='lab'>{label}</div><div class='val'>{value}</div><div class='sub'>{sub}</div></div>"


# ------------------------------------------------------------
# 5. HERO
# ------------------------------------------------------------
st.markdown("<div class='eyebrow'>Monetary policy research</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>PolicyPulse</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>Do markets move when central banks speak? This terminal studies how US and UK "
    "equity markets reacted to Federal Reserve and Bank of England decisions, what the statements actually said, "
    "and what a simple machine-learning model makes of both.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<span class='chip'><b>Event study</b> announcement-day returns</span>"
    "<span class='chip'><b>NLP</b> statement tone and topics</span>"
    "<span class='chip'><b>Machine learning</b> Random Forest scenarios</span>"
    "<span class='chip'><b>Python</b> scikit-learn, Plotly</span>",
    unsafe_allow_html=True,
)

st.write("")
choice = st.segmented_control(
    "Market", ["United States  |  FOMC", "United Kingdom  |  BoE MPC"],
    default="United States  |  FOMC", label_visibility="collapsed",
)
is_us = (choice or "United States  |  FOMC").startswith("United States")
region_key = "USA" if is_us else "UK"

with st.expander("New here? How to read this"):
    st.markdown(
        "- **Announcement-day return** is how much the stock market moved (in percent) on the day the central bank announced its decision.\n"
        "- **Event study** means comparing those days with normal days to see whether decisions move markets.\n"
        "- **Tone score** is a language-model rating of how positive a policy statement sounds, from 0 to 1.\n"
        "- **Scenario analysis** lets you change economic conditions and see what the fitted model implies. "
        "It is an illustration of the model, not a forecast."
    )


# ------------------------------------------------------------
# 6. KEY FINDINGS (computed live from the data)
# ------------------------------------------------------------
model, scaler, feats = get_model(region_key)
top_driver, top_share = None, None
if model:
    imp_all = pd.Series(model.feature_importances_, index=feats)
    top_driver, top_share = nice(imp_all.idxmax()).replace(" (%)", ""), float(imp_all.max())

st.markdown("<div class='sec' style='margin-top:22px'>Key findings</div>", unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns(4)

if is_us:
    ev = DATA["us_event"]
    sec_cols = [c for c in ev.columns if c.endswith("_Day0") and c.replace("_Day0", "") in SECTORS]
    sec_mean = ev[sec_cols].mean()
    best, worst = sec_mean.idxmax(), sec_mean.idxmin()
    sp = ev["SP500_Day0"].dropna()
    nlp = DATA["us_nlp"]
    n_pos = int((nlp["label"].astype(str).str.lower() == "positive").sum()) if "label" in nlp.columns else int((nlp["sentiment_score"] > 0.5).sum())
    f1.markdown(card("S&P 500 on FOMC days", f"{sp.mean():+.2f}%",
                     f"Average announcement-day move across {len(sp)} meetings. Up on {(sp > 0).mean() * 100:.0f}% of days, "
                     f"typical swing about {sp.std():.1f}%."), unsafe_allow_html=True)
    f2.markdown(card("Biggest sector mover", SECTORS[best.replace("_Day0", "")],
                     f"{sec_mean[best]:+.2f}% on average. {SECTORS[worst.replace('_Day0', '')]} lagged at {sec_mean[worst]:+.2f}%."),
                unsafe_allow_html=True)
    f3.markdown(card("Tone of the statements", f"{n_pos} of {len(nlp)}",
                     "FOMC statements scored positive. The Fed writes in a deliberately neutral register."),
                unsafe_allow_html=True)
    f4.markdown(card("Strongest model driver", top_driver or "n/a",
                     f"Accounts for {top_share:.0%} of the Random Forest's importance." if top_share else ""),
                unsafe_allow_html=True)
else:
    uk = DATA["uk_model"]
    hikes, cuts = int(uk["is_hike"].sum()), int(uk["is_cut"].sum())
    cut_m = uk.loc[uk["is_cut"] == 1, "FTSE_Day0"].mean() if cuts else float("nan")
    hike_m = uk.loc[uk["is_hike"] == 1, "FTSE_Day0"].mean() if hikes else float("nan")
    f1.markdown(card("Decisions studied", f"{len(uk)}",
                     f"MPC decisions from {span(uk)}: {hikes} hikes and {cuts} cuts."), unsafe_allow_html=True)
    f2.markdown(card("FTSE 100 on cut days", f"{cut_m:+.2f}%",
                     f"Average announcement-day move after a cut. After a hike it was {hike_m:+.2f}%."), unsafe_allow_html=True)
    f3.markdown(card("Typical daily swing", f"{uk['FTSE_Day0'].std():.2f}%",
                     f"Standard deviation of FTSE 100 returns on decision days. Worst day {uk['FTSE_Day0'].min():+.2f}%."),
                unsafe_allow_html=True)
    f4.markdown(card("Model caveat", "Read with care",
                     (f"{top_share:.0%} of the model's weight is on a window that includes the announcement day, "
                      "so read it as illustrative.") if top_share else ""),
                unsafe_allow_html=True)

st.write("")

tabs = st.tabs(["Rates and markets", "Announcement reactions", "Statement language", "Model", "Scenario lab", "3D explorer", "Method"])


# ------------------------------------------------------------
# TAB 1: RATES AND MARKETS
# ------------------------------------------------------------
with tabs[0]:
    section("Policy rates and yields over time",
            "Use the range buttons above the chart to zoom. Shaded bands mark major policy regimes.")
    df_macro = DATA["us_macro"] if is_us else DATA["uk_model"]
    num = df_macro.select_dtypes(include=np.number)
    cols = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
    if not is_us:
        pref = [c for c in ["BoE_Rate_Post", "BoE_Rate_Pre", "UK10Y_Current", "FTSE_CAR"] if c in cols]
        cols = pref + [c for c in cols if c not in pref]
    if cols:
        default = [c for c in (["FEDFUNDS", "DGS10"] if is_us else ["BoE_Rate_Post"]) if c in cols] or cols[:2]
        selected = st.multiselect("Series to plot", cols, default=default, format_func=nice)
        if selected:
            plot_df = df_macro[selected].rename(columns=nice)
            fig = px.line(plot_df, color_discrete_sequence=SERIES)
            fig.update_traces(connectgaps=True, line=dict(width=2.4))
            fig.update_layout(hovermode="x unified")
            if is_us:
                fig.add_vrect(x0="2020-03-15", x1="2021-12-15", fillcolor="rgba(227,168,87,0.08)", line_width=0,
                              annotation_text="Rates near zero", annotation_position="top left",
                              annotation_font=dict(size=11, color=MUTED))
                fig.add_vrect(x0="2022-03-16", x1="2023-07-26", fillcolor="rgba(224,97,90,0.08)", line_width=0,
                              annotation_text="Hiking cycle", annotation_position="top left",
                              annotation_font=dict(size=11, color=MUTED))
            else:
                fig.add_vrect(x0="2009-03-05", x1="2016-08-04", fillcolor="rgba(227,168,87,0.08)", line_width=0,
                              annotation_text="Post-crisis low rates", annotation_position="top left",
                              annotation_font=dict(size=11, color=MUTED))
            fig.update_xaxes(rangeselector=dict(
                buttons=[dict(count=1, label="1Y", step="year", stepmode="backward"),
                         dict(count=5, label="5Y", step="year", stepmode="backward"),
                         dict(count=10, label="10Y", step="year", stepmode="backward"),
                         dict(step="all", label="All")],
                bgcolor=PANEL, activecolor=BORDER, bordercolor=BORDER, font=dict(color=TEXT, size=11),
                x=1, xanchor="right", y=1.1))
            show(style(fig, height=430, legend=True))
            if is_us:
                takeaway("The Fed cut to near zero in 2020, then raised rates at the fastest pace in decades from 2022. "
                         "The 10-year yield reflects expectations about future policy, so it can move before the policy rate does.", "Reading the chart. ")
            else:
                takeaway("The Bank of England cut to 0.5% in 2009 and held near the floor for seven years. "
                         "Long stretches of unchanged rates mean few rate changes to study in that period.", "Reading the chart. ")
    else:
        st.info("No data available for this market.")


# ------------------------------------------------------------
# TAB 2: ANNOUNCEMENT REACTIONS
# ------------------------------------------------------------
with tabs[1]:
    if is_us:
        df_ev = DATA["us_event"]
        metric = "SP500_Day0" if "SP500_Day0" in df_ev.columns else (df_ev.columns[0] if len(df_ev.columns) else None)
        asset = "S&P 500"
    else:
        df_ev = DATA["uk_model"]
        metric = "FTSE_Day0"
        asset = "FTSE 100"

    section(f"How the {asset} moved on announcement days",
            "Each observation is one central bank decision. Returns are in percent.")
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
            section("Distribution of returns", "Taller means more common. The dashed line is the average.")
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(x=s, nbinsx=24, histnorm="probability density", marker_color=BLUE,
                                         marker_line_color=BG, marker_line_width=1.5, opacity=0.55,
                                         hovertemplate="Return %{x}%<extra></extra>", name="Events"))
            xs, ys = kde(s)
            if xs is not None:
                fig_h.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=ACCENT, width=2.6),
                                           hoverinfo="skip", name="Smoothed shape"))
            fig_h.add_vline(x=float(s.mean()), line_dash="dash", line_color=MUTED)
            show(style(fig_h, height=340, legend=False, x_title="Announcement-day return (%)", y_ticks=False))
        with c2:
            if is_us:
                section("Reaction by sector", "Average announcement-day move. Green gained, red fell.")
                sec_cols = [c for c in df_ev.columns if c.endswith("_Day0") and c.replace("_Day0", "") in SECTORS]
                avg = df_ev[sec_cols].mean().rename(lambda c: SECTORS[c.replace("_Day0", "")]).sort_values()
                fig_sec = go.Figure(go.Bar(
                    x=avg.values, y=avg.index, orientation="h",
                    marker_color=[UP if v >= 0 else DOWN for v in avg.values],
                    text=[f"{v:+.2f}%" for v in avg.values], textposition="outside", cliponaxis=False,
                    textfont=dict(color=TEXT, size=11),
                    hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
                ))
                fig_sec.update_xaxes(range=[min(avg.min() * 1.6, -0.05), max(avg.max() * 1.45, 0.05)], showticklabels=False)
                show(style(fig_sec, height=340, legend=False))
            else:
                section("Reaction by decision type", "Each dot is one decision.")
                kind = np.where(df_ev["is_hike"] == 1, "Hike", np.where(df_ev["is_cut"] == 1, "Cut", "Hold"))
                plot_df = pd.DataFrame({"Decision": kind, "Return (%)": df_ev[metric].values})
                order = [d for d in ["Cut", "Hold", "Hike"] if d in set(kind)]
                fig_box = px.strip(plot_df, x="Decision", y="Return (%)", color="Decision",
                                   category_orders={"Decision": order},
                                   color_discrete_map={"Hike": DOWN, "Cut": UP, "Hold": NEUTRAL})
                fig_box.update_traces(marker=dict(size=8, opacity=0.8), jitter=0.55)
                for d in order:
                    m = plot_df.loc[plot_df["Decision"] == d, "Return (%)"].mean()
                    fig_box.add_annotation(x=d, y=m, text=f"avg {m:+.2f}%", showarrow=False, xshift=62,
                                           font=dict(color=TEXT, size=11))
                show(style(fig_box, height=340, legend=False, y_title="Return (%)"))

        if is_us:
            _sm = df_ev[sec_cols].mean()
            takeaway(f"Announcement days are mostly small moves, but sectors do not react equally: "
                     f"{SECTORS[_sm.idxmax().replace('_Day0', '')]} gained most on average ({_sm.max():+.2f}%), "
                     f"while {SECTORS[_sm.idxmin().replace('_Day0', '')]} fell most ({_sm.min():+.2f}%).")
        else:
            takeaway("Returns cluster tightly around zero, suggesting most MPC decisions were largely priced in. "
                     "The outliers are the surprises.")
    else:
        st.info("No event data available for this market.")


# ------------------------------------------------------------
# TAB 3: STATEMENT LANGUAGE
# ------------------------------------------------------------
with tabs[2]:
    if is_us:
        df_nlp = DATA["us_nlp"]
        section("What the Fed says, and how it sounds",
                "Every FOMC statement was scored by a finance-tuned language model for tone and keyword emphasis.")
        if not df_nlp.empty and "sentiment_score" in df_nlp.columns:
            tone_lbl = df_nlp["label"] if "label" in df_nlp.columns else pd.Series(
                np.where(df_nlp["sentiment_score"] > 0, "positive", "neutral"), index=df_nlp.index)
            tone_df = pd.DataFrame({"Score": df_nlp["sentiment_score"], "Tone": tone_lbl.astype(str).str.capitalize()})
            n_pos = int((tone_df["Tone"] == "Positive").sum())

            section("Tone of each statement", f"{n_pos} of {len(tone_df)} statements read as clearly positive; the rest are neutral.")
            fig_tone = px.scatter(tone_df, y="Score", color="Tone",
                                  color_discrete_map={"Positive": UP, "Neutral": NEUTRAL, "Negative": DOWN})
            fig_tone.update_traces(marker=dict(size=10, line=dict(width=0), opacity=0.9))
            show(style(fig_tone, height=280, y_title="Positive-tone score"))

            topics = [c for c in ["count_inflation", "count_employment", "count_growth", "count_risk"] if c in df_nlp.columns]
            if topics:
                section("What the statements emphasise", "Keyword counts per statement, smoothed over four meetings.")
                smooth = df_nlp[topics].rolling(4, min_periods=1).mean().rename(columns=nice)
                idx = smooth.index
                gap_note = ""
                for i in range(1, len(idx)):
                    if (idx[i] - idx[i - 1]).days > 365:
                        smooth.loc[idx[i] - pd.Timedelta(days=30)] = np.nan
                        gap_note = f"The source file has no statements between {idx[i - 1]:%b %Y} and {idx[i]:%b %Y}."
                smooth = smooth.sort_index()
                fig_topic = px.line(smooth, color_discrete_sequence=SERIES)
                fig_topic.update_traces(line=dict(width=2.4))
                fig_topic.update_layout(hovermode="x unified")
                show(style(fig_topic, height=320, y_title="Mentions per statement"))
                if gap_note:
                    st.caption(gap_note)

            both = DATA["us_full"][["sentiment_score", "SP500_Day0"]]
            both = both[(both["sentiment_score"] != 0) | (both["SP500_Day0"] != 0)]
            if len(both) > 5:
                corr = both["sentiment_score"].corr(both["SP500_Day0"])
                section("Does tone matter?", f"Statement tone against the S&P 500 move that day. Correlation: {corr:+.2f}.")
                fig_sc = px.scatter(both, x="sentiment_score", y="SP500_Day0", opacity=0.8, color_discrete_sequence=[ACCENT])
                fig_sc.update_traces(marker=dict(size=9, line=dict(width=0)))
                slope, intercept = np.polyfit(both["sentiment_score"], both["SP500_Day0"], 1)
                xs = np.array([both["sentiment_score"].min(), both["sentiment_score"].max()])
                fig_sc.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                            line=dict(color=MUTED, dash="dash", width=2), name="Linear fit", hoverinfo="skip"))
                show(style(fig_sc, height=320, legend=False, x_title="Positive-tone score",
                           y_title="S&P 500 return (%)"))
                if abs(corr) < 0.3:
                    takeaway("The relationship is weak, so tone alone does not explain the market's reaction. "
                             "How the message compares with expectations likely matters more.")
                else:
                    takeaway("Tone and the market move are noticeably related in this sample, though with few "
                             "positive statements the result should be read with care.")
        else:
            st.info("No statement data available.")
    else:
        section("Statement language", "NLP analysis was built on FOMC statements only.")
        st.info("Switch to United States | FOMC above to explore the statement language analysis.")


# ------------------------------------------------------------
# TAB 4: MODEL
# ------------------------------------------------------------
with tabs[3]:
    section("What drives the model",
            "A Random Forest was fit to estimate the announcement-day return from policy and statement features. "
            "Bars show how much each input is used. This describes the model, not cause and effect.")
    if model:
        imp = pd.DataFrame({"Feature": [nice(f) for f in feats], "Importance": model.feature_importances_}).sort_values("Importance")
        fig_imp = go.Figure(go.Bar(
            x=imp["Importance"], y=imp["Feature"], orientation="h",
            marker_color=[ACCENT if i == len(imp) - 1 else BLUE for i in range(len(imp))],
            text=[f"{v:.0%}" for v in imp["Importance"]], textposition="outside", cliponaxis=False,
            textfont=dict(color=TEXT, size=12), hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig_imp.update_xaxes(showticklabels=False, range=[0, imp["Importance"].max() * 1.2])
        show(style(fig_imp, height=330, legend=False))
        extra = "" if is_us else " In the UK model, the FTSE 100 cumulative abnormal return overlaps the announcement day it is predicting, so its large weight is partly mechanical."
        takeaway("The model is fit on the full event sample and is illustrative. With under a hundred events, "
                 "it should be read as a way to explore relationships, not as a trading signal." + extra, "Caveat. ")
    else:
        st.info("Not enough data to fit the model.")


# ------------------------------------------------------------
# TAB 5: SCENARIO LAB
# ------------------------------------------------------------
PRESETS_US = {
    "Calm": dict(stress=1, inf=3, emp=4, rate=3.5, yld=3.8),
    "Inflation": dict(stress=3, inf=15, emp=4, rate=5.5, yld=4.8),
    "Stress": dict(stress=8, inf=6, emp=8, rate=2.0, yld=2.8),
}
PRESETS_UK = {
    "Hold": dict(bp=0, car=0.0),
    "+25 bp": dict(bp=25, car=0.5),
    "-50 bp": dict(bp=-50, car=-1.0),
}


def apply_preset(key, vals):
    for k, v in vals.items():
        st.session_state[f"{key}_{k}"] = v


@st.fragment
def scenario_lab():
    model, scaler, feats = get_model(region_key)
    section("Scenario lab",
            "Pick a preset or move the sliders, and see the announcement-day return the model implies "
            "compared with every past decision.")
    if not model:
        st.info("Not enough data to fit the model.")
        return

    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("<div class='lab' style='color:#8C97A9;font-size:0.72rem;font-weight:600;"
                        "text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px'>Scenario inputs</div>",
                        unsafe_allow_html=True)
            if is_us:
                for k, v in PRESETS_US["Calm"].items():
                    st.session_state.setdefault(f"us_{k}", v)
                pc = st.columns(3)
                for col, (name, vals) in zip(pc, PRESETS_US.items()):
                    col.button(name, key=f"pre_{name}", use_container_width=True, on_click=apply_preset, args=("us", vals))
                s_crash = st.slider("Market stress", 0, 10, key="us_stress",
                                    help="Higher stress lowers statement tone and raises risk language.")
                s_inf = st.slider("Inflation language", 0, 25, key="us_inf", help="Inflation mentions in the statement.")
                s_emp = st.slider("Employment language", 0, 25, key="us_emp", help="Employment mentions in the statement.")
                s_rate = st.slider("Fed funds rate (%)", 0.0, 12.0, step=0.25, key="us_rate")
                s_yield = st.slider("10-year yield (%)", 0.0, 12.0, step=0.05, key="us_yld")
                overrides = {
                    "sentiment_score": max(0.0, min(1.0, 0.5 - s_crash * 0.15)),
                    "count_risk": 5 + s_crash * 3,
                    "count_inflation": s_inf, "count_employment": s_emp,
                    "FEDFUNDS": s_rate, "DGS10": s_yield,
                }
            else:
                for k, v in PRESETS_UK["Hold"].items():
                    st.session_state.setdefault(f"uk_{k}", v)
                pc = st.columns(3)
                for col, (name, vals) in zip(pc, PRESETS_UK.items()):
                    col.button(name, key=f"pre_{name}", use_container_width=True, on_click=apply_preset, args=("uk", vals))
                s_bp = st.slider("Rate change (bp)", -100, 100, step=25, key="uk_bp")
                s_car = st.slider("FTSE 100 trend (CAR %)", -5.0, 5.0, step=0.25, key="uk_car")
                overrides = {"Rate_Change_bp": s_bp, "is_hike": 1 if s_bp > 0 else 0,
                             "is_cut": 1 if s_bp < 0 else 0, "FTSE_CAR": s_car}

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
    colr = UP if pred >= 0 else DOWN
    direction = "above" if pred >= avg_hist else "below"

    with right:
        st.markdown(
            f"<div class='result'><div class='lab'>Model-implied announcement-day return</div>"
            f"<div class='big' style='color:{colr}'>{pred:+.2f}%</div>"
            f"<div class='sub' style='color:{MUTED};margin-top:10px;font-size:0.9rem'>"
            f"{abs(pred - avg_hist):.2f} points {direction} the historical average of {avg_hist:+.2f}%. "
            f"Higher than {pct:.0f}% of past decisions.</div>"
            f"<div class='pbar'><div class='pdot' style='left:{min(max(pct, 2), 98)}%;background:{colr}'></div></div>"
            f"<div class='pticks'><span>Weakest day in history</span><span>Strongest</span></div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(x=hist, nbinsx=24, histnorm="probability density", marker_color=BLUE,
                                     marker_line_color=BG, marker_line_width=1.5, opacity=0.5,
                                     hovertemplate="Return %{x}%<extra></extra>"))
        xs, ys = kde(hist)
        if xs is not None:
            fig_h.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=MUTED, width=2), hoverinfo="skip"))
        fig_h.add_vline(x=pred, line_color=colr, line_width=3, annotation_text="scenario",
                        annotation_position="top", annotation_font=dict(color=colr, size=12))
        show(style(fig_h, height=250, legend=False, x_title="Historical announcement-day returns (%)", y_ticks=False))

    with st.expander("Inputs used by the model"):
        compare = pd.DataFrame({
            "Input": [nice(f) for f in feats],
            "Latest in data": [round(float(latest_vals[f]), 2) for f in feats],
            "Scenario": [round(float(base[f]), 2) for f in feats],
        })
        st.dataframe(compare, hide_index=True, use_container_width=True)


with tabs[4]:
    scenario_lab()


# ------------------------------------------------------------
# TAB 6: 3D EXPLORER
# ------------------------------------------------------------
with tabs[5]:
    section("3D explorer", "Pick any three variables to see how they relate. Drag to rotate, scroll to zoom.")
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
        fig_3d = px.scatter_3d(df_3d, x=x_ax, y=y_ax, z=z_ax, color=z_ax, opacity=0.85,
                               color_continuous_scale="Viridis",
                               labels={x_ax: nice(x_ax), y_ax: nice(y_ax), z_ax: nice(z_ax)})
        fig_3d.update_traces(marker=dict(size=4.5))
        ax = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.07)",
                  zerolinecolor="rgba(255,255,255,0.12)", color=MUTED)
        fig_3d.update_layout(scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, bgcolor="rgba(0,0,0,0)"), coloraxis_showscale=False)
        show(style(fig_3d, height=600, legend=False))
    else:
        st.info("Not enough variables to plot.")


# ------------------------------------------------------------
# TAB 7: METHOD
# ------------------------------------------------------------
with tabs[6]:
    section("Method, data and limits")
    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.markdown("**Question.** Do central bank decisions and the words around them move equity markets, "
                    "and can a simple model capture that relationship?")
        st.markdown("**Event study.** For each FOMC and MPC decision, equity returns on the announcement day and the "
                    "cumulative abnormal return around it are measured, including sector breakdowns for the US.")
        st.markdown("**Statement language.** FOMC statements from 2010 onward are scored with a finance-tuned language model "
                    "for tone, and keyword counts track emphasis on inflation, employment, growth and risk.")
        st.markdown("**Model.** A Random Forest regressor (150 trees, depth 6) relates policy and language features to the "
                    "announcement-day return. The scenario lab feeds it hypothetical inputs.")
    with m2:
        st.markdown("**Data.** Federal Reserve Economic Data (FRED), FOMC statements, Bank of England policy rates, "
                    "and index and sector prices from Yahoo Finance.")
        st.markdown("**Limits.** Small samples (about 90 US and 70 UK events), no out-of-sample test, and the model is fit "
                    "on all events. Results are illustrative and not a trading signal.")
        st.markdown("**Built with.** Python, pandas, NumPy, scikit-learn, Plotly and Streamlit.")
        st.markdown(f"**Source code.** [{REPO_URL.replace('https://', '')}]({REPO_URL})")
        st.markdown("**Team.** Devangi Agrahari and Bhanu Teja Dunna, MS in Finance (Quantitative Finance), University at Buffalo.")


st.markdown("<hr style='margin-top:2.4rem'>", unsafe_allow_html=True)
st.markdown(
    "<div class='note'>PolicyPulse is for academic and research purposes only and does not constitute investment advice.</div>",
    unsafe_allow_html=True,
)
