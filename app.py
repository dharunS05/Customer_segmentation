"""
app.py — Streamlit dashboard for Customer Segmentation
Run: streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import load_and_clean, build_rfm
from train import (
    engineer_features,
    find_optimal_k,
    train,
    save_artifacts,
    load_artifacts,
    profile_clusters,
    SEGMENT_MAP,
    SEGMENT_COLORS,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* Main background */
.main {background: #f8fafc;}

/* KPI cards */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
    border-left: 4px solid;
}
.kpi-value {font-size: 2rem; font-weight: 700; line-height: 1.1; color: #0f172a;}
.kpi-label {font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin-top: 4px;}
.kpi-sub   {font-size: 0.75rem; color: #94a3b8; margin-top: 2px;}

/* Section header */
.section-header {
    font-size: 1rem; font-weight: 600; color: #0f172a;
    margin: 28px 0 12px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px;
}

/* Segment badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .03em;
}

/* Upload area hint */
.upload-hint {
    font-size: 0.75rem; color: #94a3b8; margin-top: 8px;
    text-align: center; line-height: 1.5;
}

/* Scrollable table wrapper */
.table-wrap {overflow-x: auto;}

/* Hide Streamlit default footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────
def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Customer Segmentation")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Dataset",
        type=["csv"],
        help="Upload the Online Retail II CSV file",
    )

    st.markdown("##### Model Settings")
    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4, step=1)

    run_btn = st.button("▶  Run Segmentation", use_container_width=True)

    # Load existing results if available
    st.markdown("---")
    st.markdown("##### Saved Results")
    if os.path.exists("data/final_customer_segments.csv"):
        if st.button("📂 Load Previous Results", use_container_width=True):
            st.session_state["rfm_labeled"] = pd.read_csv("data/final_customer_segments.csv")
            st.session_state["profile"] = profile_clusters(st.session_state["rfm_labeled"])
            st.session_state["ready"] = True
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div class='upload-hint'>Built for learning<br>Unsupervised ML · RFM · KMeans</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Run pipeline on button click
# ──────────────────────────────────────────────
if run_btn and uploaded is not None:
    with st.spinner("Processing data and training model…"):
        try:
            # Save upload temporarily
            os.makedirs("data", exist_ok=True)
            raw_path = "data/_upload_temp.csv"
            with open(raw_path, "wb") as f:
                f.write(uploaded.read())

            df_clean = load_and_clean(raw_path)
            rfm = build_rfm(df_clean)
            rfm_labeled, model, scaler = train(rfm, n_clusters=n_clusters)

            os.makedirs("models", exist_ok=True)
            save_artifacts(model, scaler)

            rfm_labeled.to_csv("data/final_customer_segments.csv", index=False)

            # Elbow / silhouette data
            X_scaled, _, _ = engineer_features(rfm)
            k_vals, inertia, sil_scores = find_optimal_k(X_scaled, range(2, 11))

            st.session_state.update({
                "rfm_labeled": rfm_labeled,
                "profile": profile_clusters(rfm_labeled),
                "k_vals": k_vals,
                "inertia": inertia,
                "sil_scores": sil_scores,
                "ready": True,
            })
            st.success("✅ Segmentation complete!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif run_btn and uploaded is None:
    st.sidebar.warning("⚠️ Please upload a CSV file first.")


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────
if not st.session_state.get("ready"):
    # ── Landing / welcome ──
    st.markdown("# 🎯 Customer Segmentation Dashboard")
    st.markdown("##### Unsupervised ML · RFM Analysis · KMeans Clustering")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📥 Step 1 — Upload**
        Upload the *Online Retail II* dataset (CSV) from the sidebar.
        """)
    with col2:
        st.markdown("""
        **⚙️ Step 2 — Configure**
        Choose the number of clusters (default: 4) using the slider.
        """)
    with col3:
        st.markdown("""
        **▶ Step 3 — Run**
        Click *Run Segmentation* to clean data, build RFM features, and train KMeans.
        """)

    st.info("💡 **Dataset:** [Online Retail II — UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)")
    st.stop()


# ──────────────────────────────────────────────
# Data is ready — render dashboard
# ──────────────────────────────────────────────
rfm = st.session_state["rfm_labeled"]
profile = st.session_state["profile"]

# ── Title ──
st.markdown("# 🎯 Customer Segmentation Dashboard")
st.markdown("---")

# ── KPI Row ──
k1, k2, k3, k4 = st.columns(4)
total_customers = rfm["customer_id"].nunique()
total_revenue   = rfm["monetary"].sum()
avg_recency     = rfm["recency"].mean()
avg_frequency   = rfm["frequency"].mean()

def kpi_card(col, value, label, sub, color):
    col.markdown(f"""
    <div class="kpi-card" style="border-color:{color}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(k1, f"{total_customers:,}",  "Total Customers",    "Unique segmented",     "#4A90D9")
kpi_card(k2, f"£{total_revenue:,.0f}","Total Revenue",      "Sum of all purchases", "#F4A836")
kpi_card(k3, f"{avg_recency:.0f}d",   "Avg. Recency",       "Days since last order","#E05C5C")
kpi_card(k4, f"{avg_frequency:.1f}",  "Avg. Purchase Freq.","Orders per customer",  "#5CB85C")

st.markdown("<br>", unsafe_allow_html=True)

# ── Segment Overview ──
st.markdown("<div class='section-header'>Segment Distribution</div>", unsafe_allow_html=True)

col_pie, col_profile = st.columns([1, 1.5], gap="large")

with col_pie:
    colors = [SEGMENT_COLORS.get(s, "#aaa") for s in profile["segment_name"]]
    fig_pie = px.pie(
        profile,
        values="customers",
        names="segment_name",
        color="segment_name",
        color_discrete_map=SEGMENT_COLORS,
        hole=0.45,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_profile:
    st.markdown("**Cluster Profile Summary**")
    display_profile = profile[["segment_name", "customers", "avg_recency", "avg_frequency", "avg_monetary"]].copy()
    display_profile.columns = ["Segment", "Customers", "Avg Recency (days)", "Avg Frequency", "Avg Monetary (£)"]
    st.dataframe(display_profile, use_container_width=True, hide_index=True)

    # Segment description cards
    st.markdown("<br>", unsafe_allow_html=True)
    descriptions = {
        "VIP Loyalists":          ("🏆", "High value, frequent buyers. Top priority for retention."),
        "Repeat Value Buyers":    ("🔄", "Consistent buyers with moderate value. Nurture to VIP."),
        "Active Growth Customers":("🚀", "Recently active. Potential to grow with right offers."),
        "At-Risk Low Value":      ("⚠️", "Inactive and low spend. Re-engagement needed."),
    }
    for seg, (icon, desc) in descriptions.items():
        color = SEGMENT_COLORS.get(seg, "#aaa")
        st.markdown(
            f"<span style='color:{color};font-weight:700'>{icon} {seg}</span> — "
            f"<span style='color:#475569;font-size:.85rem'>{desc}</span>",
            unsafe_allow_html=True,
        )

# ── RFM Feature Charts ──
st.markdown("<div class='section-header'>RFM Feature Analysis</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

def rfm_box(col, feature, title):
    fig = px.box(
        rfm, x="segment_name", y=feature,
        color="segment_name", color_discrete_map=SEGMENT_COLORS,
        labels={"segment_name": "", feature: title},
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=30, b=60, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        xaxis_tickangle=-20,
        font=dict(size=11),
    )
    col.plotly_chart(fig, use_container_width=True)

rfm_box(c1, "recency",   "Recency (days)")
rfm_box(c2, "frequency", "Frequency (orders)")
rfm_box(c3, "monetary",  "Monetary (£)")

# ── Bar Chart — Avg Revenue per Segment ──
st.markdown("<div class='section-header'>Average Revenue by Segment</div>", unsafe_allow_html=True)

fig_bar = px.bar(
    profile.sort_values("avg_monetary", ascending=False),
    x="segment_name", y="avg_monetary",
    color="segment_name",
    color_discrete_map=SEGMENT_COLORS,
    labels={"segment_name": "Segment", "avg_monetary": "Avg Revenue (£)"},
    text_auto=".0f",
)
fig_bar.update_layout(
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f8fafc",
    margin=dict(t=10, b=10),
    font=dict(size=12),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Elbow / Silhouette (only when freshly trained) ──
if "k_vals" in st.session_state:
    st.markdown("<div class='section-header'>Model Evaluation — Elbow & Silhouette</div>", unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)

    with ec1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=st.session_state["k_vals"],
            y=st.session_state["inertia"],
            mode="lines+markers",
            marker=dict(size=8, color="#4A90D9"),
            line=dict(color="#4A90D9", width=2),
        ))
        fig_elbow.update_layout(
            title="Elbow Method (Inertia)",
            xaxis_title="K", yaxis_title="Inertia",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f8fafc",
            margin=dict(t=40, b=20),
        )
        ec1.plotly_chart(fig_elbow, use_container_width=True)

    with ec2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=st.session_state["k_vals"],
            y=st.session_state["sil_scores"],
            mode="lines+markers",
            marker=dict(size=8, color="#5CB85C"),
            line=dict(color="#5CB85C", width=2),
        ))
        fig_sil.update_layout(
            title="Silhouette Score by K",
            xaxis_title="K", yaxis_title="Silhouette Score",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f8fafc",
            margin=dict(t=40, b=20),
        )
        ec2.plotly_chart(fig_sil, use_container_width=True)

# ── 3D Scatter ──
st.markdown("<div class='section-header'>3D RFM Cluster View</div>", unsafe_allow_html=True)

fig_3d = px.scatter_3d(
    rfm.sample(min(3000, len(rfm)), random_state=42),
    x="recency", y="frequency", z="monetary",
    color="segment_name",
    color_discrete_map=SEGMENT_COLORS,
    opacity=0.6,
    labels={"recency": "Recency", "frequency": "Frequency", "monetary": "Monetary"},
    hover_data=["customer_id"],
)
fig_3d.update_layout(
    margin=dict(t=20, b=10, l=10, r=10),
    paper_bgcolor="rgba(0,0,0,0)",
    scene=dict(bgcolor="#f8fafc"),
    legend=dict(title="Segment"),
)
st.plotly_chart(fig_3d, use_container_width=True)

# ── Full Customer Table ──
st.markdown("<div class='section-header'>Customer Segment Table</div>", unsafe_allow_html=True)

seg_filter = st.multiselect(
    "Filter by Segment",
    options=sorted(rfm["segment_name"].unique()),
    default=sorted(rfm["segment_name"].unique()),
)

filtered = rfm[rfm["segment_name"].isin(seg_filter)]

display_cols = ["customer_id", "recency", "frequency", "monetary", "segment_name"]
st.dataframe(
    filtered[display_cols].rename(columns={
        "customer_id": "Customer ID",
        "recency": "Recency (days)",
        "frequency": "Frequency",
        "monetary": "Monetary (£)",
        "segment_name": "Segment",
    }),
    use_container_width=True,
    hide_index=True,
    height=340,
)

# Download
csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️  Download Filtered Results (CSV)",
    data=csv_bytes,
    file_name="customer_segments.csv",
    mime="text/csv",
)
