"""
dashboard/app.py
================
Streamlit AML Compliance Dashboard.

Pages
-----
1. Executive Overview  – KPIs, time-series anomaly trends
2. Anomaly Explorer    – Transaction-level alert table + VAE score histogram
3. Network Graph       – Interactive NetworkX visualisation of transaction clusters
4. SAR Reports         – Full SAR viewer + PDF/JSON export
5. Model Performance   – Confusion matrix, P/R/F1, training history curves
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AML Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MODEL_DIR = Path("models")

# ---------------------------------------------------------------------------
# Cached data loaders
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> str:
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/security-checked.png", width=64
    )
    st.sidebar.title("AML Command Center")
    st.sidebar.caption("AI-Powered Anti-Money Laundering")
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navigate",
        options=[
            "📊 Executive Overview",
            "🔎 Anomaly Explorer",
            "🕸️ Network Graph",
            "📋 SAR Reports",
            "📈 Model Performance",
            "🔬 Causal What-If",
            "🦆 DuckDB Analytics",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("Data freshness: auto-refresh every 5 min.")
    return page


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_executive_overview() -> None:
    st.title("📊 Executive Overview")
    txn_df = load_transactions()
    cust_df = load_customers()
    sar_df = load_sar_summary()

    if txn_df.empty:
        st.warning("No transaction data found. Run `generate_data.py` first.")
        return

    total_txn = len(txn_df)
    total_suspicious = int(txn_df["is_suspicious"].sum())
    total_customers = len(cust_df)
    total_sars = len(sar_df)
    flagged_amount = txn_df.loc[txn_df["is_suspicious"], "amount_usd"].sum()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{total_txn:,}")
    c2.metric("Suspicious Alerts", f"{total_suspicious:,}", delta=f"{100*total_suspicious/total_txn:.2f}%")
    c3.metric("Customers Monitored", f"{total_customers:,}")
    c4.metric("SARs Generated", f"{total_sars:,}")
    c5.metric("Flagged Volume (USD)", f"${flagged_amount:,.0f}")

    st.divider()

    # Time-series anomaly trend
    st.subheader("Daily Suspicious Transaction Volume")
    daily = (
        txn_df[txn_df["is_suspicious"]]
        .set_index("timestamp")
        .resample("D")["amount_usd"]
        .agg(["count", "sum"])
        .rename(columns={"count": "Alert Count", "sum": "Flagged Amount (USD)"})
    )
    if not daily.empty:
        tab1, tab2 = st.tabs(["Alert Count", "Flagged Amount"])
        with tab1:
            st.area_chart(daily["Alert Count"], use_container_width=True)
        with tab2:
            st.area_chart(daily["Flagged Amount (USD)"], use_container_width=True)

    st.divider()

    # Typology breakdown
    st.subheader("Suspicious Transaction Typology Breakdown")
    if "label" in txn_df.columns:
        typo_counts = (
            txn_df[txn_df["is_suspicious"]]["label"]
            .value_counts()
            .reset_index()
        )
        typo_counts.columns = ["Typology", "Count"]
        col_l, col_r = st.columns(2)
        col_l.dataframe(typo_counts, use_container_width=True)
        col_r.bar_chart(typo_counts.set_index("Typology"))

    # Jurisdiction risk heatmap
    if not cust_df.empty and "jurisdiction_risk" in cust_df.columns:
        st.subheader("Customer Jurisdiction Risk Distribution")
        jur_counts = cust_df["jurisdiction_risk"].value_counts().reset_index()
        jur_counts.columns = ["Jurisdiction Risk", "Count"]
        st.bar_chart(jur_counts.set_index("Jurisdiction Risk"))


def page_anomaly_explorer() -> None:
    st.title("🔎 Anomaly Explorer")
    txn_df = load_transactions()
    if txn_df.empty:
        st.warning("No data available.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_only_flagged = st.checkbox("Show flagged only", value=True)
    with col2:
        min_amount = st.number_input("Min amount (USD)", min_value=0.0, value=0.0, step=100.0)
    with col3:
        typologies = ["All"] + sorted(txn_df["label"].unique().tolist()) if "label" in txn_df.columns else ["All"]
        selected_typo = st.selectbox("Typology", typologies)

    filtered = txn_df.copy()
    if show_only_flagged:
        filtered = filtered[filtered["is_suspicious"]]
    if min_amount > 0:
        filtered = filtered[filtered["amount_usd"] >= min_amount]
    if selected_typo != "All" and "label" in filtered.columns:
        filtered = filtered[filtered["label"] == selected_typo]

    st.markdown(f"**{len(filtered):,} transactions** match filters.")

    display_cols = [
        "transaction_id", "timestamp", "sender_id", "receiver_id",
        "amount_usd", "transaction_type", "label", "is_suspicious",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].sort_values("amount_usd", ascending=False).head(500),
        use_container_width=True,
    )

    # Download
    csv = filtered.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "aml_alerts.csv", "text/csv")

    st.divider()
    st.subheader("Amount Distribution – Flagged vs Normal")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    normal = txn_df[~txn_df["is_suspicious"]]["amount_usd"].clip(upper=txn_df["amount_usd"].quantile(0.99))
    suspicious = txn_df[txn_df["is_suspicious"]]["amount_usd"].clip(upper=txn_df["amount_usd"].quantile(0.99))
    ax.hist(normal, bins=80, alpha=0.55, color="#2196F3", label="Normal", density=True)
    ax.hist(suspicious, bins=80, alpha=0.65, color="#F44336", label="Suspicious", density=True)
    ax.set_xlabel("Transaction Amount (USD)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Amount Distribution by Label")
    st.pyplot(fig, use_container_width=True)
    plt.close()


def page_network_graph() -> None:
    st.title("🕸️ Transaction Network Graph")

    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        st.error("networkx and matplotlib are required for this page.")
        return

    txn_df = load_transactions()
    risk_df = load_risk_scores()

    if txn_df.empty:
        st.warning("No transaction data loaded.")
        return

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider("Edges to visualise", 200, 5000, 1000, 200)
    with col2:
        layout_algo = st.selectbox("Layout algorithm", ["spring", "kamada_kawai", "circular"])

    sample = txn_df.sample(min(sample_size, len(txn_df)), random_state=42)

    G = nx.DiGraph()
    for _, row in sample.iterrows():
        src, dst = row["sender_id"], row["receiver_id"]
        for node in [src, dst]:
            if not G.has_node(node):
                G.add_node(node)
        G.add_edge(src, dst, weight=float(row["amount_usd"]))

    # Colour nodes by GNN risk score if available
    if not risk_df.empty and "gnn_risk_score" in risk_df.columns:
        risk_map = risk_df.set_index("customer_id")["gnn_risk_score"].to_dict()
        node_colors = [risk_map.get(n, 0.0) for n in G.nodes()]
    else:
        node_colors = [0.3] * G.number_of_nodes()

    st.info(f"Displaying {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    with st.spinner("Rendering graph …"):
        fig, ax = plt.subplots(figsize=(12, 8))
        layout_fn = {
            "spring": lambda g: nx.spring_layout(g, k=0.5, seed=42),
            "kamada_kawai": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
        }[layout_algo]
        pos = layout_fn(G)
        node_sizes = [20 + 200 * G.degree(n) / max(dict(G.degree()).values(), default=1) for n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes,
            node_color=node_colors, cmap=cm.RdYlGn_r, alpha=0.8, ax=ax,
        )
        nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="#555", ax=ax, arrows=False)
        ax.set_title("Customer Transaction Network (colour = GNN risk score)", fontsize=13)
        ax.axis("off")
        sm = plt.cm.ScalarMappable(cmap=cm.RdYlGn_r, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="GNN Risk Score")
    st.pyplot(fig, use_container_width=True)
    plt.close()


def page_sar_reports() -> None:
    st.title("📋 SAR Reports")
    sar_df = load_sar_summary()
    sar_list = load_sar_reports()

    if sar_df.empty:
        st.warning("No SAR reports generated yet. Run `agents/report_writer.py`.")
        return

    # Summary table
    st.subheader(f"SAR Summary – {len(sar_df)} Reports")
    priority_filter = st.multiselect(
        "Filter by priority", options=["CRITICAL", "HIGH", "MEDIUM"], default=["CRITICAL", "HIGH"]
    )
    filtered_df = sar_df[sar_df["priority"].isin(priority_filter)] if priority_filter else sar_df
    st.dataframe(filtered_df, use_container_width=True)

    # JSON download
    st.download_button(
        "⬇ Download SAR JSON",
        data=json.dumps(sar_list, indent=2, default=str),
        file_name="sar_reports.json",
        mime="application/json",
    )
    st.download_button(
        "⬇ Download SAR CSV",
        data=sar_df.to_csv(index=False).encode(),
        file_name="sar_summary.csv",
        mime="text/csv",
    )

    st.divider()

    # Individual SAR viewer
    st.subheader("Individual SAR Viewer")
    if sar_list:
        selected_idx = st.selectbox(
            "Select SAR",
            range(len(sar_list)),
            format_func=lambda i: f"{sar_list[i].get('sar_id','?')} – {sar_list[i].get('priority','?')}",
        )
        sar = sar_list[selected_idx]
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Filer Information**")
            st.json(sar.get("filer", {}))
        with col_r:
            st.markdown("**Suspect Information**")
            st.json(sar.get("suspect", {}))
        st.markdown("**Activity Description**")
        activity = sar.get("activity", {})
        st.info(activity.get("activity_description", "No narrative available."))
        with st.expander("Full Activity JSON"):
            st.json(activity)


def page_model_performance() -> None:
    st.title("📈 Model Performance")
    meta = load_vae_meta()
    history = load_vae_history()

    if not meta:
        st.warning("Model metadata not found. Train the VAE first.")
        return

    # VAE metrics
    st.subheader("VAE – Configuration & Thresholds")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Input Dim", meta.get("input_dim", "–"))
    mcol2.metric("Latent Dim", meta.get("latent_dim", "–"))
    mcol3.metric("β (KL weight)", meta.get("beta", "–"))
    mcol4.metric("Anomaly Threshold", f"{meta.get('threshold', 0):.6f}")

    if not history.empty:
        st.subheader("VAE Training History")
        st.line_chart(history.set_index(history.index)[["loss", "recon", "kl"]])

    # Confusion matrix
    if "confusion_matrix" in meta:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        cm_data = np.array(meta["confusion_matrix"])
        st.subheader("VAE Confusion Matrix (Test Set)")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_data, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Suspicious"])
        ax.set_yticklabels(["Normal", "Suspicious"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_data[i, j]), ha="center", va="center", color="black", fontsize=14)
        plt.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close()

    # GNN training history
    gnn_hist_path = REPORTS_DIR / "gnn_training_history.csv"
    if gnn_hist_path.exists():
        gnn_hist = pd.read_csv(gnn_hist_path)
        st.subheader("GNN Training History")
        st.line_chart(gnn_hist.set_index("epoch")[["train_loss", "val_recall"]])


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    page = sidebar()
    route_map = {
        "📊 Executive Overview": page_executive_overview,
        "🔎 Anomaly Explorer": page_anomaly_explorer,
        "🕸️ Network Graph": page_network_graph,
        "📋 SAR Reports": page_sar_reports,
        "📈 Model Performance": page_model_performance,
    }
    route_map[page]()


if __name__ == "__main__":
    main()