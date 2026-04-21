"""
dashboard/app.py
================
APEX AML Intelligence Platform — Production-grade Streamlit dashboard.

Design: Dark industrial / military-grade ops center aesthetic.
Font: JetBrains Mono for data, Barlow Condensed for headers.
Color: Deep navy + electric cyan accent + amber alert + red critical.

Pages
-----
  Command Center   — Real-time KPIs, live anomaly feed, threat gauge
  Threat Analysis  — VAE/GAN score distributions, SHAP explainability
  Network Intel    — Transaction graph, cluster analysis
  SAR Operations   — Full SAR lifecycle viewer and exporter
  Model Registry   — Training curves, confusion matrix, performance metrics
  Policy Simulator — Causal what-if threshold and rule analysis
  Data Intelligence — DuckDB SQL analytics engine
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="APEX AML Intelligence Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark ops-center aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&family=Barlow:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #050912 !important;
    color: #c8d6e5 !important;
}
.stApp {
    background: linear-gradient(135deg, #050912 0%, #080e1f 50%, #050912 100%) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080e1f !important;
    border-right: 1px solid #0d1f3c !important;
}
section[data-testid="stSidebar"] * { color: #8ba3c0 !important; }
section[data-testid="stSidebar"] .stRadio label { 
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    padding: 6px 0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 100%) !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #0066cc);
}
[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #4a7fa5 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a3a5c !important;
    border-radius: 4px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00d4ff !important;
    color: #00d4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 2px !important;
    padding: 0.4rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #00d4ff !important;
    color: #050912 !important;
}

/* ── Select boxes / inputs ── */
.stSelectbox > div > div, .stTextInput > div > div {
    background: #0a1628 !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 2px !important;
    color: #c8d6e5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Sliders ── */
.stSlider > div > div { background: #1a3a5c !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a3a5c !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #4a7fa5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: #0a1628 !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #0a1628 !important;
    border: 1px solid #1a3a5c !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #4a7fa5 !important;
}

/* ── Info / warning / error boxes ── */
.stInfo { background: #051828 !important; border-left: 3px solid #00d4ff !important; }
.stWarning { background: #1a1200 !important; border-left: 3px solid #f5a623 !important; }
.stError { background: #1a0505 !important; border-left: 3px solid #ff3366 !important; }
.stSuccess { background: #051a0d !important; border-left: 3px solid #00ff88 !important; }

/* ── Dividers ── */
hr { border-color: #1a3a5c !important; margin: 1rem 0 !important; }

/* ── Code blocks ── */
code { 
    font-family: 'JetBrains Mono', monospace !important;
    background: #0a1628 !important;
    border: 1px solid #1a3a5c !important;
    padding: 2px 6px !important;
    border-radius: 2px !important;
    color: #00d4ff !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR   = ROOT / "models"

# Add project root to path so internal imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Matplotlib dark theme
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    "figure.facecolor":  "#080e1f",
    "axes.facecolor":    "#0a1628",
    "axes.edgecolor":    "#1a3a5c",
    "axes.labelcolor":   "#8ba3c0",
    "axes.titlecolor":   "#c8d6e5",
    "text.color":        "#8ba3c0",
    "xtick.color":       "#4a7fa5",
    "ytick.color":       "#4a7fa5",
    "grid.color":        "#1a3a5c",
    "grid.alpha":        0.6,
    "lines.color":       "#00d4ff",
    "patch.edgecolor":   "#1a3a5c",
    "legend.facecolor":  "#0a1628",
    "legend.edgecolor":  "#1a3a5c",
    "legend.labelcolor": "#8ba3c0",
    "font.family":       "monospace",
})

CYAN    = "#00d4ff"
AMBER   = "#f5a623"
RED     = "#ff3366"
GREEN   = "#00ff88"
NAVY    = "#0a1628"
MUTED   = "#4a7fa5"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_transactions() -> pd.DataFrame:
    p = DATA_DIR / "transactions.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_customers() -> pd.DataFrame:
    p = DATA_DIR / "customers.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_sar_summary() -> pd.DataFrame:
    p = REPORTS_DIR / "sar_summary.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_sar_reports() -> list:
    p = REPORTS_DIR / "sar_reports.json"
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_data(ttl=300, show_spinner=False)
def load_risk_scores() -> pd.DataFrame:
    p = REPORTS_DIR / "customer_risk_scores.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_vae_history() -> pd.DataFrame:
    p = REPORTS_DIR / "vae_training_history.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_vae_meta() -> dict:
    p = MODEL_DIR / "vae_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data(ttl=300, show_spinner=False)
def load_shap_importance() -> pd.DataFrame:
    p = REPORTS_DIR / "shap_feature_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_vae_alerts() -> pd.DataFrame:
    p = REPORTS_DIR / "vae_alerts.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;padding-bottom:0.75rem;border-bottom:1px solid #1a3a5c;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                    color:#4a7fa5;letter-spacing:0.2em;text-transform:uppercase;
                    margin-bottom:0.25rem;">APEX AML Platform</div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                    font-weight:700;color:#e8f4ff;letter-spacing:0.02em;">{title}</div>
        {f'<div style="font-family:Barlow,sans-serif;font-size:0.85rem;color:#4a7fa5;margin-top:0.2rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, level: str) -> str:
    colors = {
        "CRITICAL": ("#ff3366", "#1a0510"),
        "HIGH":     ("#f5a623", "#1a0f00"),
        "MEDIUM":   ("#00d4ff", "#001828"),
        "LOW":      ("#00ff88", "#001a0d"),
        "ACTIVE":   ("#00ff88", "#001a0d"),
        "DRAFT":    ("#4a7fa5", "#080e1f"),
    }
    fg, bg = colors.get(level.upper(), ("#8ba3c0", "#0a1628"))
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg}33;'
        f'font-family:JetBrains Mono,monospace;font-size:0.65rem;letter-spacing:0.1em;'
        f'padding:2px 8px;border-radius:2px;text-transform:uppercase;">{text}</span>'
    )


def data_missing_notice(message: str) -> None:
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1a3a5c;border-left:3px solid #4a7fa5;
                padding:1.5rem;border-radius:2px;margin:1rem 0;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.1em;margin-bottom:0.5rem;">
            NO DATA AVAILABLE
        </div>
        <div style="font-family:'Barlow',sans-serif;font-size:0.9rem;color:#8ba3c0;">
            {message}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#2a5a7c;margin-top:0.75rem;">
            RUN: python main.py pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> str:
    with st.sidebar:
        # Logo block
        st.markdown("""
        <div style="padding:1.5rem 0 1rem 0;border-bottom:1px solid #1a3a5c;margin-bottom:1rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;
                        font-weight:700;color:#00d4ff;letter-spacing:0.05em;">
                APEX AML
            </div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.75rem;
                        color:#2a5a7c;letter-spacing:0.2em;text-transform:uppercase;">
                Intelligence Platform v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        # System status
        txn_exists   = (DATA_DIR / "transactions.parquet").exists()
        vae_exists   = (MODEL_DIR / "vae_model.pth").exists()
        gnn_exists   = (MODEL_DIR / "gnn_model.pth").exists()
        gan_exists   = (MODEL_DIR / "gan_discriminator.pth").exists()
        sar_exists   = (REPORTS_DIR / "sar_summary.csv").exists()

        def dot(ok: bool) -> str:
            c = "#00ff88" if ok else "#ff3366"
            return f'<span style="color:{c};font-size:0.65rem;">◆</span>'

        st.markdown(f"""
        <div style="background:#070d1a;border:1px solid #1a3a5c;border-radius:2px;
                    padding:0.75rem;margin-bottom:1rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                        color:#2a5a7c;letter-spacing:0.15em;margin-bottom:0.5rem;">
                SYSTEM STATUS
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        display:flex;flex-direction:column;gap:4px;">
                <div>{dot(txn_exists)} &nbsp; DATA ENGINE</div>
                <div>{dot(vae_exists)} &nbsp; VAE DETECTOR</div>
                <div>{dot(gan_exists)} &nbsp; GAN FUSION</div>
                <div>{dot(gnn_exists)} &nbsp; GNN INVESTIGATOR</div>
                <div>{dot(sar_exists)} &nbsp; SAR GENERATOR</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                    color:#2a5a7c;letter-spacing:0.15em;margin-bottom:0.5rem;">
            NAVIGATION
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            label="",
            options=[
                "Command Center",
                "Threat Analysis",
                "Network Intelligence",
                "SAR Operations",
                "Model Registry",
                "Policy Simulator",
                "Data Intelligence",
            ],
            label_visibility="collapsed",
        )

        st.markdown("""
        <div style="position:fixed;bottom:1rem;left:1rem;right:1rem;
                    font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                    color:#1a3a5c;text-align:center;">
            APEX AML © 2026 — CONFIDENTIAL
        </div>
        """, unsafe_allow_html=True)

    return page


# ---------------------------------------------------------------------------
# PAGE 1: Command Center
# ---------------------------------------------------------------------------
def page_command_center() -> None:
    page_header(
        "Command Center",
        "Real-time threat monitoring and operational overview"
    )

    txn_df  = load_transactions()
    cust_df = load_customers()
    sar_df  = load_sar_summary()

    if txn_df.empty:
        data_missing_notice("No transaction data. Run the pipeline to generate data.")
        return

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total_txn   = len(txn_df)
    total_susp  = int(txn_df["is_suspicious"].sum())
    flag_rate   = total_susp / max(total_txn, 1) * 100
    total_vol   = txn_df["amount_usd"].sum()
    flagged_vol = txn_df.loc[txn_df["is_suspicious"], "amount_usd"].sum()
    total_sars  = len(sar_df)
    crit_sars   = len(sar_df[sar_df["priority"] == "CRITICAL"]) if not sar_df.empty and "priority" in sar_df.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Transactions Monitored", f"{total_txn:,}")
    c2.metric("Alerts Triggered",       f"{total_susp:,}", delta=f"{flag_rate:.2f}% flag rate")
    c3.metric("Total Volume (USD)",     f"${total_vol/1e6:.1f}M")
    c4.metric("Flagged Volume (USD)",   f"${flagged_vol/1e6:.1f}M")
    c5.metric("SARs Generated",         f"{total_sars:,}")
    c6.metric("Critical Alerts",        f"{crit_sars:,}", delta="Needs review" if crit_sars > 0 else "Clear")

    st.markdown("---")

    # ── Time-series anomaly trend ─────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            ANOMALY ACTIVITY TIMELINE
        </div>
        """, unsafe_allow_html=True)

        # FIXED: removed the problematic flagged_vol lambda; only total and suspicious needed for the plot
        daily = (
            txn_df.set_index("timestamp")
            .resample("W")
            .agg(
                total=("amount_usd", "count"),
                suspicious=("is_suspicious", "sum"),
            )
            .fillna(0)
        )

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(daily.index, daily["suspicious"],
                        alpha=0.15, color=CYAN)
        ax.plot(daily.index, daily["suspicious"],
                color=CYAN, linewidth=1.5, label="Suspicious Alerts")
        ax.set_xlabel("")
        ax.set_ylabel("Alert Count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            TYPOLOGY DISTRIBUTION
        </div>
        """, unsafe_allow_html=True)

        if "label" in txn_df.columns:
            typo = txn_df[txn_df["is_suspicious"]]["label"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
            colors_pie = [CYAN, AMBER, RED, GREEN, MUTED][:len(typo)]
            wedges, texts, autotexts = ax2.pie(
                typo.values, labels=typo.index,
                colors=colors_pie, autopct="%1.0f%%",
                startangle=90, pctdistance=0.75,
                wedgeprops={"linewidth": 1.5, "edgecolor": "#080e1f"},
            )
            for t in texts:     t.set_fontsize(7); t.set_color("#8ba3c0")
            for t in autotexts: t.set_fontsize(7); t.set_color("#050912"); t.set_fontweight("bold")
            ax2.set_facecolor("#0a1628")
            plt.tight_layout(pad=0.3)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

    st.markdown("---")

    # ── Latest alerts table ───────────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
        RECENT HIGH-VALUE ALERTS
    </div>
    """, unsafe_allow_html=True)

    recent_alerts = (
        txn_df[txn_df["is_suspicious"]]
        .sort_values("amount_usd", ascending=False)
        .head(15)[["transaction_id","timestamp","sender_id","receiver_id",
                   "amount_usd","transaction_type","label","country_origin","country_dest"]]
    )
    recent_alerts["amount_usd"] = recent_alerts["amount_usd"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(
        recent_alerts.reset_index(drop=True),
        use_container_width=True,
        height=350,
    )


# ---------------------------------------------------------------------------
# PAGE 2: Threat Analysis
# ---------------------------------------------------------------------------
def page_threat_analysis() -> None:
    page_header(
        "Threat Analysis",
        "VAE reconstruction error analysis, GAN fusion scores, and SHAP explainability"
    )

    txn_df  = load_transactions()
    meta    = load_vae_meta()
    alerts  = load_vae_alerts()
    shap_df = load_shap_importance()

    if txn_df.empty:
        data_missing_notice("Run the pipeline to generate transactions and train the VAE model.")
        return

    tab1, tab2, tab3 = st.tabs(["Score Distribution", "SHAP Explainability", "Alert Deep-Dive"])

    # ── Tab 1: Score distributions ────────────────────────────────────────────
    with tab1:
        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
        col_meta1.metric("VAE Threshold",   f"{meta.get('threshold', 0):.6f}")
        col_meta2.metric("Latent Dim",       meta.get("latent_dim", "—"))
        col_meta3.metric("Beta (KL Weight)", meta.get("beta", "—"))
        col_meta4.metric("GAN Fusion",
                         "ACTIVE" if meta.get("gan_fusion_active") else "INACTIVE")

        if not alerts.empty and "vae_score" in alerts.columns:
            # Merge with suspicious labels
            susp_map = txn_df.groupby("sender_id")["is_suspicious"].max()
            alerts["is_suspicious"] = alerts["customer_id"].map(susp_map).fillna(False)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # VAE score distribution
            ax = axes[0]
            normal_scores  = alerts.loc[~alerts["is_suspicious"], "vae_score"]
            susp_scores    = alerts.loc[alerts["is_suspicious"], "vae_score"]
            bins = np.linspace(0, np.percentile(alerts["vae_score"], 99), 60)
            ax.hist(normal_scores,  bins=bins, alpha=0.55, color=CYAN,  label="Normal",     density=True)
            ax.hist(susp_scores,    bins=bins, alpha=0.70, color=RED,   label="Suspicious", density=True)
            if meta.get("threshold"):
                ax.axvline(meta["threshold"], color=AMBER, linestyle="--", linewidth=1.5,
                           label=f"Threshold {meta['threshold']:.4f}")
            ax.set_title("VAE Reconstruction Error", fontsize=9, color="#c8d6e5")
            ax.set_xlabel("Anomaly Score", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

            # Combined score distribution
            if "combined_score" in alerts.columns:
                ax2 = axes[1]
                norm2 = alerts.loc[~alerts["is_suspicious"], "combined_score"]
                susp2 = alerts.loc[alerts["is_suspicious"], "combined_score"]
                bins2 = np.linspace(0, 1, 60)
                ax2.hist(norm2, bins=bins2, alpha=0.55, color=CYAN, label="Normal",     density=True)
                ax2.hist(susp2, bins=bins2, alpha=0.70, color=RED,  label="Suspicious", density=True)
                ax2.set_title("VAE+GAN Combined Score", fontsize=9, color="#c8d6e5")
                ax2.set_xlabel("Combined Anomaly Score", fontsize=8)
                ax2.set_ylabel("Density", fontsize=8)
                ax2.legend(fontsize=7)
                ax2.tick_params(labelsize=7)

            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Train the VAE model to see score distributions.")

    # ── Tab 2: SHAP ───────────────────────────────────────────────────────────
    with tab2:
        if shap_df.empty:
            st.markdown("""
            <div style="background:#0a1628;border:1px solid #1a3a5c;border-left:3px solid #f5a623;
                        padding:1.5rem;border-radius:2px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#f5a623;margin-bottom:0.5rem;">SHAP NOT YET COMPUTED</div>
                <div style="font-family:'Barlow',sans-serif;font-size:0.85rem;color:#8ba3c0;">
                    SHAP explainability runs automatically during VAE training.
                    Ensure <code>shap</code> is installed: <code>pip install shap</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:#4a7fa5;letter-spacing:0.12em;margin-bottom:1rem;">
                FEATURE IMPORTANCE — MEAN ABSOLUTE SHAP VALUE
            </div>
            """, unsafe_allow_html=True)

            col_shap, col_table = st.columns([2, 1])
            with col_shap:
                fig, ax = plt.subplots(figsize=(8, 4))
                top = shap_df.head(9)
                y_pos = range(len(top))
                bars = ax.barh(
                    list(y_pos), top["shap_importance"],
                    color=[CYAN if i == 0 else MUTED for i in range(len(top))],
                    height=0.65, edgecolor="#1a3a5c",
                )
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(top["feature"], fontsize=8)
                ax.set_xlabel("Mean |SHAP| Value", fontsize=8)
                ax.set_title("Why Transactions Get Flagged", fontsize=9, color="#c8d6e5")
                ax.tick_params(labelsize=7)
                ax.invert_yaxis()
                for bar in bars:
                    ax.text(
                        bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                        f"{bar.get_width():.4f}", va="center", ha="left",
                        fontsize=7, color="#4a7fa5",
                    )
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col_table:
                st.dataframe(
                    shap_df.rename(columns={"shap_importance": "SHAP Score"}),
                    use_container_width=True, height=300,
                )

            # SHAP summary PNG if it exists
            shap_img = REPORTS_DIR / "shap_summary.png"
            if shap_img.exists():
                st.markdown("""
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a7fa5;letter-spacing:0.12em;margin:1rem 0 0.5rem 0;">
                    SHAP BEESWARM — SAMPLE-LEVEL FEATURE IMPACT
                </div>
                """, unsafe_allow_html=True)
                st.image(str(shap_img), use_column_width=True)

    # ── Tab 3: Alert Deep-Dive ────────────────────────────────────────────────
    with tab3:
        if not alerts.empty:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
                TOP ANOMALOUS CUSTOMERS BY COMBINED SCORE
            </div>
            """, unsafe_allow_html=True)

            score_col = "combined_score" if "combined_score" in alerts.columns else "vae_score"
            top_alerts = alerts.nlargest(50, score_col)
            st.dataframe(top_alerts.reset_index(drop=True), use_container_width=True, height=400)

            # Score distribution over time
            if not txn_df.empty:
                st.markdown("""
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a7fa5;letter-spacing:0.12em;margin:1rem 0 0.5rem 0;">
                    TRANSACTION AMOUNT VS ANOMALY FLAG
                </div>
                """, unsafe_allow_html=True)
                sample = txn_df.sample(min(5000, len(txn_df)), random_state=42)
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.scatter(
                    sample.loc[~sample["is_suspicious"], "timestamp"],
                    sample.loc[~sample["is_suspicious"], "amount_usd"],
                    alpha=0.05, s=3, color=MUTED, label="Normal",
                )
                ax.scatter(
                    sample.loc[sample["is_suspicious"], "timestamp"],
                    sample.loc[sample["is_suspicious"], "amount_usd"],
                    alpha=0.4, s=6, color=RED, label="Flagged",
                )
                ax.set_ylabel("Amount (USD)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)
                ax.set_yscale("log")
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()


# ---------------------------------------------------------------------------
# PAGE 3: Network Intelligence
# ---------------------------------------------------------------------------
def page_network_intelligence() -> None:
    page_header(
        "Network Intelligence",
        "GNN-based customer risk classification and transaction network topology"
    )

    txn_df   = load_transactions()
    risk_df  = load_risk_scores()

    if txn_df.empty:
        data_missing_notice("Run the pipeline to generate data and train the GNN.")
        return

    col_cfg, col_main = st.columns([1, 3])

    with col_cfg:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            GRAPH PARAMETERS
        </div>
        """, unsafe_allow_html=True)
        sample_size  = st.slider("Edges to render", 300, 5000, 1500, 100)
        layout_algo  = st.selectbox("Layout", ["spring", "kamada_kawai", "circular"], index=0)
        gnn_threshold = st.slider("GNN risk threshold", 0.0, 1.0, 0.5, 0.05)

        if not risk_df.empty:
            high_risk_n = len(risk_df[risk_df.get("gnn_risk_score", pd.Series([])) > gnn_threshold])
            st.metric("High-Risk Nodes", f"{high_risk_n:,}")

    with col_main:
        try:
            import networkx as nx
            import matplotlib.cm as cm

            sample = txn_df.sample(min(sample_size, len(txn_df)), random_state=42)
            G = nx.DiGraph()
            for _, row in sample.iterrows():
                src, dst = row["sender_id"], row["receiver_id"]
                G.add_edge(src, dst, weight=float(row["amount_usd"]))

            # Node colours from GNN risk scores
            if not risk_df.empty and "gnn_risk_score" in risk_df.columns:
                risk_map = risk_df.set_index("customer_id")["gnn_risk_score"].to_dict()
                node_colors = [risk_map.get(n, 0.15) for n in G.nodes()]
            else:
                node_colors = [0.2] * G.number_of_nodes()

            fig, ax = plt.subplots(figsize=(11, 7))
            layout_fn = {
                "spring":       lambda g: nx.spring_layout(g, k=0.6, seed=42, iterations=30),
                "kamada_kawai": nx.kamada_kawai_layout,
                "circular":     nx.circular_layout,
            }[layout_algo]

            with st.spinner("Computing network layout…"):
                pos = layout_fn(G)

            degrees    = dict(G.degree())
            max_degree = max(degrees.values()) if degrees else 1
            node_sizes = [15 + 180 * degrees.get(n, 0) / max_degree for n in G.nodes()]

            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=cm.RdYlGn_r,
                vmin=0, vmax=1,
                alpha=0.85,
            )
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                alpha=0.06, edge_color="#4a7fa5",
                arrows=False, width=0.5,
            )
            ax.set_title(
                f"Transaction Network — {G.number_of_nodes():,} nodes, "
                f"{G.number_of_edges():,} edges",
                fontsize=9, color="#c8d6e5",
            )
            ax.axis("off")
            sm = plt.cm.ScalarMappable(cmap=cm.RdYlGn_r, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
            cb.set_label("GNN Risk Score", fontsize=7, color="#8ba3c0")
            cb.ax.tick_params(labelsize=6, colors="#4a7fa5")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        except ImportError:
            st.error("networkx not installed. Run: pip install networkx")

    # GNN risk score distribution
    if not risk_df.empty and "gnn_risk_score" in risk_df.columns:
        st.markdown("---")
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            GNN CUSTOMER RISK SCORE DISTRIBUTION
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(risk_df["gnn_risk_score"], bins=60, color=CYAN, alpha=0.7, edgecolor="#1a3a5c")
        ax.axvline(gnn_threshold, color=AMBER, linewidth=1.5, linestyle="--",
                   label=f"Threshold {gnn_threshold:.2f}")
        ax.set_xlabel("GNN Risk Score", fontsize=8)
        ax.set_ylabel("Customer Count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ---------------------------------------------------------------------------
# PAGE 4: SAR Operations
# ---------------------------------------------------------------------------
def page_sar_operations() -> None:
    page_header(
        "SAR Operations",
        "Suspicious Activity Report lifecycle — view, filter, export"
    )

    sar_df   = load_sar_summary()
    sar_list = load_sar_reports()

    if sar_df.empty:
        data_missing_notice("No SARs generated. Run: python main.py pipeline --steps write_reports")
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    total    = len(sar_df)
    critical = len(sar_df[sar_df["priority"] == "CRITICAL"]) if "priority" in sar_df.columns else 0
    high     = len(sar_df[sar_df["priority"] == "HIGH"])     if "priority" in sar_df.columns else 0
    llm_used = sar_df["llm_used"].mode()[0] if "llm_used" in sar_df.columns and not sar_df["llm_used"].isna().all() else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total SARs",      f"{total:,}")
    c2.metric("Critical",        f"{critical:,}")
    c3.metric("High Priority",   f"{high:,}")
    c4.metric("LLM Engine",      llm_used)

    st.markdown("---")

    # ── Filter + table ────────────────────────────────────────────────────────
    filter_col, _ = st.columns([2, 3])
    with filter_col:
        priorities = st.multiselect(
            "Filter by priority",
            options=["CRITICAL", "HIGH", "MEDIUM"],
            default=["CRITICAL", "HIGH"],
            label_visibility="visible",
        )

    filtered = sar_df[sar_df["priority"].isin(priorities)] if priorities and "priority" in sar_df.columns else sar_df

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.5rem;">
        SHOWING {len(filtered):,} REPORTS
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=300)

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "Download SAR JSON",
        data=json.dumps(sar_list, indent=2, default=str),
        file_name="sar_reports.json",
        mime="application/json",
    )
    col_dl2.download_button(
        "Download SAR CSV",
        data=sar_df.to_csv(index=False).encode(),
        file_name="sar_summary.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Individual SAR viewer ─────────────────────────────────────────────────
    if sar_list:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            INDIVIDUAL SAR VIEWER
        </div>
        """, unsafe_allow_html=True)

        idx = st.selectbox(
            "Select report",
            range(len(sar_list)),
            format_func=lambda i: (
                f"[{sar_list[i].get('priority','?'):<8}] "
                f"{sar_list[i].get('sar_id','?')}  —  "
                f"{sar_list[i].get('suspect',{}).get('customer_id','?')}"
            ),
            label_visibility="collapsed",
        )
        sar = sar_list[idx]

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                        color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.5rem;">
                SUSPECT PROFILE
            </div>
            """, unsafe_allow_html=True)
            suspect = sar.get("suspect", {})
            for k, v in suspect.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            border-bottom:1px solid #1a3a5c;padding:4px 0;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                 color:#4a7fa5;">{k.upper()}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 color:#c8d6e5;">{str(v)[:40]}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_r:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                        color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.5rem;">
                ACTIVITY SUMMARY
            </div>
            """, unsafe_allow_html=True)
            activity = sar.get("activity", {})
            for k, v in activity.items():
                if k == "activity_description":
                    continue
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            border-bottom:1px solid #1a3a5c;padding:4px 0;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                 color:#4a7fa5;">{k.upper()}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 color:#c8d6e5;">{str(v)[:40]}</span>
                </div>
                """, unsafe_allow_html=True)

        narrative = activity.get("activity_description", "")
        if narrative:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                        color:#4a7fa5;letter-spacing:0.12em;margin:1rem 0 0.5rem;">
                LLM-GENERATED NARRATIVE
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #1a3a5c;border-left:3px solid #00d4ff;
                        padding:1rem;font-family:'Barlow',sans-serif;font-size:0.85rem;
                        color:#c8d6e5;line-height:1.6;border-radius:2px;">
                {narrative}
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 5: Model Registry
# ---------------------------------------------------------------------------
def page_model_registry() -> None:
    page_header(
        "Model Registry",
        "Training history, performance metrics, and model configuration"
    )

    meta    = load_vae_meta()
    history = load_vae_history()

    if not meta:
        data_missing_notice("No trained models found. Run: python main.py pipeline --steps train_vae")
        return

    # ── Config row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Input Dimension",  meta.get("input_dim", "—"))
    c2.metric("Latent Dimension", meta.get("latent_dim", "—"))
    c3.metric("Beta (KL)",        meta.get("beta", "—"))
    c4.metric("Anomaly Threshold", f"{meta.get('threshold', 0):.6f}")
    c5.metric("GAN Fusion",       "Active" if meta.get("gan_fusion_active") else "Off")

    st.markdown("---")

    if not history.empty:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            VAE TRAINING CONVERGENCE
        </div>
        """, unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, (col, color, label) in zip(axes, [
            ("loss",  CYAN,  "Total ELBO Loss"),
            ("recon", AMBER, "Reconstruction Loss"),
            ("kl",    GREEN, "KL Divergence"),
        ]):
            if col in history.columns:
                ax.plot(history[col], color=color, linewidth=1.5)
                ax.fill_between(range(len(history)), history[col], alpha=0.1, color=color)
                ax.set_title(label, fontsize=8, color="#c8d6e5")
                ax.set_xlabel("Epoch", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if "confusion_matrix" in meta:
        st.markdown("---")
        col_cm, col_metrics = st.columns([1, 2])

        with col_cm:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
                CONFUSION MATRIX
            </div>
            """, unsafe_allow_html=True)
            cm = np.array(meta["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Normal", "Suspicious"], fontsize=8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Normal", "Suspicious"], fontsize=8)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            thresh = cm.max() / 2
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "#c8d6e5",
                            fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_metrics:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
                DERIVED METRICS
            </div>
            """, unsafe_allow_html=True)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                precision  = tp / max(tp + fp, 1)
                recall     = tp / max(tp + fn, 1)
                f1         = 2 * precision * recall / max(precision + recall, 1e-9)
                specificity = tn / max(tn + fp, 1)
                for label, val, color in [
                    ("Precision",    f"{precision:.4f}",   CYAN),
                    ("Recall",       f"{recall:.4f}",      GREEN),
                    ("F1 Score",     f"{f1:.4f}",          AMBER),
                    ("Specificity",  f"{specificity:.4f}", MUTED),
                    ("True Positives",  str(tp), CYAN),
                    ("False Positives", str(fp), AMBER),
                    ("False Negatives", str(fn), RED),
                ]:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                border-bottom:1px solid #1a3a5c;padding:6px 0;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                     color:#4a7fa5;">{label}</span>
                        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;
                                     font-weight:700;color:{color};">{val}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ── GNN training history ──────────────────────────────────────────────────
    gnn_hist_path = REPORTS_DIR / "gnn_training_history.csv"
    if gnn_hist_path.exists():
        gnn_hist = pd.read_csv(gnn_hist_path)
        st.markdown("---")
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            GNN TRAINING HISTORY
        </div>
        """, unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        if "train_loss" in gnn_hist.columns:
            axes[0].plot(gnn_hist["epoch"], gnn_hist["train_loss"], color=CYAN, linewidth=1.5)
            axes[0].set_title("Focal Loss", fontsize=8, color="#c8d6e5")
            axes[0].tick_params(labelsize=7)
        if "val_recall" in gnn_hist.columns:
            axes[1].plot(gnn_hist["epoch"], gnn_hist["val_recall"], color=GREEN, linewidth=1.5)
            axes[1].axhline(0.90, color=AMBER, linestyle="--", linewidth=1, label="Target 0.90")
            axes[1].set_title("Validation Recall", fontsize=8, color="#c8d6e5")
            axes[1].tick_params(labelsize=7)
            axes[1].legend(fontsize=7)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ---------------------------------------------------------------------------
# PAGE 6: Policy Simulator (Causal What-If)
# ---------------------------------------------------------------------------
def page_policy_simulator() -> None:
    page_header(
        "Policy Simulator",
        "Causal inference and counterfactual what-if analysis for AML rule evaluation"
    )

    txn_path = DATA_DIR / "transactions.parquet"
    if not txn_path.exists():
        data_missing_notice("No transaction data. Run: python main.py pipeline --steps generate_data")
        return

    # Safe causal import
    try:
        sys.path.insert(0, str(ROOT))
        from causal.causal_inference import AMLCausalAnalyzer
        causal_available = True
    except ModuleNotFoundError:
        causal_available = False
    except Exception:
        causal_available = False

    if not causal_available:
        st.markdown("""
        <div style="background:#0a1628;border:1px solid #f5a623;border-left:3px solid #f5a623;
                    padding:1.5rem;border-radius:2px;margin-bottom:1rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:#f5a623;margin-bottom:0.5rem;">MODULE PATH FIX REQUIRED</div>
            <div style="font-family:'Barlow',sans-serif;font-size:0.85rem;color:#8ba3c0;">
                The causal module needs to be on the Python path. Run the dashboard from the 
                project root directory:<br><br>
                <code>cd C:\\...\\AI_business_platform</code><br>
                <code>streamlit run dashboard/app.py</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Still show the UI with stub data
        _policy_simulator_stub()
        return

    txn_df = pd.read_parquet(txn_path)
    txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
    analyzer = AMLCausalAnalyzer(txn_df)

    meta             = load_vae_meta()
    current_threshold = meta.get("threshold", 0.01)

    tab1, tab2, tab3 = st.tabs(["Threshold Simulation", "Rule Impact", "Causal Effect (DiD)"])

    with tab1:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            DETECTION THRESHOLD COUNTERFACTUAL ANALYSIS
        </div>
        """, unsafe_allow_html=True)

        col_ctrl, col_res = st.columns([1, 2])
        with col_ctrl:
            st.metric("Current Threshold", f"{current_threshold:.6f}")
            pct_change = st.slider("Threshold change (%)", -60, 60, -20, 5)
            new_thresh = current_threshold * (1 + pct_change / 100)
            st.metric("New Threshold", f"{new_thresh:.6f}")
            run_sim = st.button("Run Simulation")

        with col_res:
            if run_sim:
                with st.spinner("Simulating counterfactual…"):
                    result = analyzer.what_if_threshold(current_threshold, new_thresh)
                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline Alerts",  f"{result.baseline_alerts:,}")
                c2.metric("New Alert Volume", f"{result.counterfactual_alerts:,}",
                          delta=f"{result.delta_pct:+.1f}%")
                c3.metric("Flagged $ Change",
                          f"${result.counterfactual_flagged_amount - result.baseline_flagged_amount:+,.0f}")
                st.markdown(f"""
                <div style="background:#0a1628;border:1px solid #1a3a5c;border-left:3px solid #00d4ff;
                            padding:1rem;margin-top:1rem;font-family:'Barlow',sans-serif;
                            font-size:0.85rem;color:#c8d6e5;line-height:1.6;">
                    {result.recommendation}
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            NEW DETECTION RULE IMPACT ESTIMATION
        </div>
        """, unsafe_allow_html=True)

        rule_name     = st.text_input("Rule description",
                                      value="Flag cross-border wire transfers >$8K to FATF high-risk jurisdictions")
        col_a, col_b  = st.columns(2)
        affected_frac = col_a.slider("Fraction of transactions affected (%)", 1, 40, 15) / 100
        uplift_pct    = col_b.slider("Expected detection uplift (%)", 1, 50, 18)

        if st.button("Estimate Rule Impact"):
            with st.spinner("Estimating…"):
                rr = analyzer.what_if_rule(rule_name, affected_frac, float(uplift_pct))
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Baseline",    f"{rr.baseline_alerts:,}")
            rc2.metric("With Rule",   f"{rr.counterfactual_alerts:,}", delta=f"+{rr.delta_alerts:,}")
            rc3.metric("Alert Uplift", f"{rr.delta_pct:+.1f}%")
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #1a3a5c;border-left:3px solid #00ff88;
                        padding:1rem;margin-top:1rem;font-family:'Barlow',sans-serif;
                        font-size:0.85rem;color:#c8d6e5;">
                {rr.recommendation}
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
            DIFFERENCE-IN-DIFFERENCES CAUSAL EFFECT ESTIMATION
        </div>
        """, unsafe_allow_html=True)

        col_date, col_pol = st.columns(2)
        policy_date = col_date.date_input("Policy date", value=pd.Timestamp("2023-07-01"))
        policy_name = col_pol.text_input("Policy name", value="Enhanced Jurisdiction Screening")

        if st.button("Estimate Causal Effect"):
            with st.spinner("Running DiD estimation…"):
                effect = analyzer.estimate_rule_effect(
                    rule_name=policy_name,
                    policy_date=str(policy_date),
                    treated_column="structuring_flag",
                )
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Estimator",       effect.estimator)
            ec2.metric("ATE",             f"{effect.ate:+.4f}")
            ec3.metric("Relative Effect", f"{effect.relative_effect_pct:+.1f}%")
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #1a3a5c;border-left:3px solid #f5a623;
                        padding:1rem;margin-top:1rem;font-family:'Barlow',sans-serif;
                        font-size:0.85rem;color:#c8d6e5;">
                {effect.interpretation}
                <br><br>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4a7fa5;">
                95% CI: [{effect.ate_lower:.4f}, {effect.ate_upper:.4f}]
                </span>
            </div>
            """, unsafe_allow_html=True)


def _policy_simulator_stub() -> None:
    """Show simulator UI with stub values when causal module unavailable."""
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
        THRESHOLD SIMULATION — PREVIEW MODE
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    col1.slider("Threshold change (%)", -60, 60, -20, 5)
    col2.markdown("""
    <div style="background:#0a1628;border:1px solid #1a3a5c;padding:1rem;margin-top:1rem;
                font-family:'Barlow',sans-serif;font-size:0.85rem;color:#4a7fa5;">
        Run the dashboard from the project root directory to enable live simulation.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 7: Data Intelligence (DuckDB)
# ---------------------------------------------------------------------------
def page_data_intelligence() -> None:
    page_header(
        "Data Intelligence",
        "High-performance SQL analytics over the full Parquet dataset via DuckDB"
    )

    # Safe DuckDB import
    duckdb_available = False
    engine = None
    try:
        sys.path.insert(0, str(ROOT))
        from data.duckdb_queries import AMLQueryEngine
        txn_path = DATA_DIR / "transactions.parquet"
        if txn_path.exists():
            engine = AMLQueryEngine()
            duckdb_available = True
    except ModuleNotFoundError:
        pass
    except ImportError:
        pass
    except Exception as e:
        st.error(f"DuckDB init error: {e}")

    if not duckdb_available or engine is None:
        st.markdown("""
        <div style="background:#0a1628;border:1px solid #f5a623;border-left:3px solid #f5a623;
                    padding:1.5rem;border-radius:2px;margin-bottom:1.5rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:#f5a623;margin-bottom:0.5rem;">INSTALL DUCKDB TO ENABLE</div>
            <div style="font-family:'Barlow',sans-serif;font-size:0.85rem;color:#8ba3c0;">
                Install duckdb and generate data first:
            </div>
            <code>pip install duckdb</code><br>
            <code>python main.py pipeline --steps generate_data</code>
        </div>
        """, unsafe_allow_html=True)

        # Show pre-computed CSVs if they exist
        csv_files = list(REPORTS_DIR.glob("duckdb_*.csv"))
        if csv_files:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:#4a7fa5;letter-spacing:0.12em;margin-bottom:0.75rem;">
                PRE-COMPUTED ANALYTICS (FROM LAST PIPELINE RUN)
            </div>
            """, unsafe_allow_html=True)
            selected = st.selectbox(
                "Select report",
                csv_files,
                format_func=lambda p: p.stem.replace("duckdb_", "").replace("_", " ").upper(),
                label_visibility="collapsed",
            )
            df = pd.read_csv(selected)
            st.dataframe(df, use_container_width=True, height=400)
        return

    # ── Live DuckDB UI ────────────────────────────────────────────────────────
    preset_queries = {
        "Transaction Summary":       engine.transaction_summary,
        "Top Suspicious Customers":  lambda: engine.top_suspicious_customers(30),
        "Daily Alert Trend (90d)":   lambda: engine.daily_alert_trend(90),
        "Typology Breakdown":        engine.typology_breakdown,
        "Cross-Border Corridors":    engine.cross_border_analysis,
        "Structuring Alerts":        engine.structuring_alerts,
        "Hourly Velocity Anomalies": engine.hourly_velocity_anomalies,
    }

    col_sel, col_run = st.columns([3, 1])
    with col_sel:
        selected_query = st.selectbox(
            "Preset query",
            list(preset_queries.keys()),
            label_visibility="collapsed",
        )
    with col_run:
        run_btn = st.button("Execute Query")

    # Custom SQL expander
    with st.expander("Custom SQL"):
        custom_sql = st.text_area(
            "SQL",
            height=100,
            placeholder="SELECT sender_id, COUNT(*) AS cnt FROM transactions WHERE is_suspicious = TRUE GROUP BY sender_id ORDER BY cnt DESC LIMIT 20",
            label_visibility="collapsed",
        )
        run_custom = st.button("Run SQL")

    if run_btn:
        with st.spinner("Querying…"):
            try:
                df = preset_queries[selected_query]()
                st.markdown(f"""
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                            color:#00ff88;margin-bottom:0.5rem;">
                    RETURNED {len(df):,} ROWS
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, height=400)

                # Auto-chart numeric results
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                str_cols = df.select_dtypes(exclude="number").columns.tolist()
                if numeric_cols and str_cols and len(df) > 1:
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    x_col = str_cols[0]
                    y_col = numeric_cols[0]
                    top_n = df.head(15)
                    ax.bar(range(len(top_n)), top_n[y_col], color=CYAN, alpha=0.8, edgecolor="#1a3a5c")
                    ax.set_xticks(range(len(top_n)))
                    ax.set_xticklabels(top_n[x_col].astype(str), rotation=45, ha="right", fontsize=7)
                    ax.set_ylabel(y_col, fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.set_title(selected_query, fontsize=9, color="#c8d6e5")
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False).encode(),
                    file_name=f"aml_{selected_query.lower().replace(' ','_')}.csv",
                )
            except Exception as exc:
                st.error(f"Query error: {exc}")

    if run_custom and custom_sql.strip():
        with st.spinner("Running custom SQL…"):
            try:
                df = engine.sql(custom_sql)
                st.dataframe(df, use_container_width=True, height=300)
            except Exception as exc:
                st.error(f"SQL error: {exc}")

    if engine:
        engine.close()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main() -> None:
    page = sidebar()
    routes = {
        "Command Center":       page_command_center,
        "Threat Analysis":      page_threat_analysis,
        "Network Intelligence": page_network_intelligence,
        "SAR Operations":       page_sar_operations,
        "Model Registry":       page_model_registry,
        "Policy Simulator":     page_policy_simulator,
        "Data Intelligence":    page_data_intelligence,
    }
    routes[page]()


if __name__ == "__main__":
    main()