"""
app.py — Streamlit Deployment
Customer Conversion Prediction Engine
ZINB-Based E-Commerce Analytics Tool

Project: Zero-Inflated Count Modeling of Customer Purchases in E-Commerce Platforms
Student: VISHNU B | Reg: 24MSKR0023 | MSc Data Science 2024-26
Guide: ROHINI S NAIR
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import io
from conversion_engine import (
    predict_customer,
    score_dataframe,
    assign_segment,
    RECOMMENDATIONS,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ZINB Conversion Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 0.85rem;
        opacity: 0.7;
        margin: 0;
        font-family: 'IBM Plex Mono', monospace;
    }

    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6c757d;
        margin-bottom: 0.3rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 600;
        color: #212529;
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: #868e96;
        margin-top: 0.3rem;
    }

    .segment-badge {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        margin: 0.5rem 0;
    }

    .recommendation-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .formula-box {
        background: #1e1e2e;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        margin: 1rem 0;
    }

    .section-header {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #868e96;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #dee2e6;
        font-family: 'IBM Plex Mono', monospace;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #4361ee, #7209b7);
    }

    div[data-testid="stTabs"] button {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = "zinb_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f), True
    else:
        # Return mock model for demo purposes
        return _build_mock_model(), False


def _build_mock_model():
    """Demo model when zinb_model.pkl is not found."""
    class MockZINB:
        params = {"alpha": 0.67}

        def predict(self, X, which="mean"):
            lv = X["log_views"].values
            lc = X["log_carts"].values
            if which == "prob-main":
                return np.clip(0.92 - 0.12 * lc - 0.03 * lv, 0.02, 0.98)
            elif which == "lin":
                return 0.25 + 0.45 * lc + 0.08 * lv
    return MockZINB()


model, model_loaded = load_model()

# ─────────────────────────────────────────────
# SEGMENT COLORS
# ─────────────────────────────────────────────

SEGMENT_CONFIG = {
    "Cold User":           {"color": "#6c757d", "bg": "#f8f9fa", "icon": "❄️"},
    "Warm Lead":           {"color": "#e67700", "bg": "#fff9db", "icon": "🔥"},
    "Potential Buyer":     {"color": "#0c7abf", "bg": "#e7f5ff", "icon": "🎯"},
    "High Value Customer": {"color": "#2b8a3e", "bg": "#ebfbee", "icon": "⭐"},
}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📊 Customer Conversion - Prediction Engine</h1>
    <p>Zero-Inflated Negative Binomial (ZINB) Model · MSc Data Science Project · VISHNU B · 24MSKR0023</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning(
        "⚠️ **Demo Mode**: `zinb_model.pkl` not found. "
        "Running with a simulated model for interface demonstration. "
        "Replace with your trained model file to use real predictions.",
        icon="⚠️"
    )

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Customer", "📂 Batch Scoring", "📘 Model Reference", "📈 Diagnostics"])

# ═══════════════════════════════════════════════
# TAB 1: SINGLE CUSTOMER PREDICTION
# ═══════════════════════════════════════════════

with tab1:
    col_input, col_output = st.columns([1, 1.6], gap="large")

    with col_input:
        st.markdown('<div class="section-header">Customer Inputs</div>', unsafe_allow_html=True)

        total_views = st.number_input(
            "Total Page Views",
            min_value=0,
            max_value=100_000,
            value=50,
            step=1,
            help="Total number of product pages viewed by this customer."
        )

        total_carts = st.number_input(
            "Total Cart Additions",
            min_value=0,
            max_value=10_000,
            value=3,
            step=1,
            help="Total number of times customer added items to cart."
        )

        predict_btn = st.button("▶ Predict", type="primary", use_container_width=True)

        st.markdown('<div class="section-header">Feature Transform</div>', unsafe_allow_html=True)
        log_views = np.log1p(total_views)
        log_carts = np.log1p(total_carts)

        st.markdown(f"""
        <div class="formula-box">
            log_views = log(1 + {total_views}) = {log_views:.4f}<br>
            log_carts = log(1 + {total_carts}) = {log_carts:.4f}
        </div>
        """, unsafe_allow_html=True)

        st.caption("Log1p transform applied to match training pipeline.")

    with col_output:
        if predict_btn or True:  # Always show output, refresh on button
            profile = predict_customer(model, int(total_views), int(total_carts))
            seg_cfg = SEGMENT_CONFIG[profile.segment]

            st.markdown('<div class="section-header">ZINB Model Outputs</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">P(Non-Buyer)</div>
                    <div class="metric-value">{profile.p_zero:.3f}</div>
                    <div class="metric-sub">Structural zero prob.</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">E(Purchases)</div>
                    <div class="metric-value">{profile.mu:.3f}</div>
                    <div class="metric-sub">Given potential buyer</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Dispersion α</div>
                    <div class="metric-value">{profile.alpha:.3f}</div>
                    <div class="metric-sub">NB dispersion param</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Conversion Score</div>', unsafe_allow_html=True)

            score_pct = min(profile.conversion_score / 5.0, 1.0)  # normalize for bar (cap at 5)
            st.markdown(f"""
            <div class="formula-box">
                Score = (1 − p) × μ = (1 − {profile.p_zero:.3f}) × {profile.mu:.3f}
                      = <strong style="color:#a6e3a1">{profile.conversion_score:.4f}</strong>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score_pct, text=f"Score: {profile.conversion_score:.4f}")
            if profile.p_zero > 0.85:
                st.info("This customer is highly likely to be a structural non-buyer.")
            elif profile.p_zero > 0.50:
                st.info("This customer shows moderate purchase hesitation.")
            else:
                st.success("This customer is likely part of the active buyer population.")
            st.markdown('<div class="section-header">Segment & Recommendation</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="
                background: {seg_cfg['bg']};
                border: 1.5px solid {seg_cfg['color']};
                border-radius: 10px;
                padding: 1.2rem 1.5rem;
            ">
                <div style="
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: {seg_cfg['color']};
                    margin-bottom: 0.6rem;
                ">
                    {seg_cfg['icon']}  {profile.segment}
                </div>
                <div style="font-size: 0.88rem; color: #343a40; line-height: 1.65;">
                    {profile.recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 2: BATCH SCORING
# ═══════════════════════════════════════════════

with tab2:
    st.markdown(
        '<div class="section-header">Batch Customer Scoring</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Upload a CSV with columns **`total_views`** and **`total_carts`** "
        "(optional: any other columns are preserved). "
        "The engine will score every row and return a downloadable scored file."
    )

    col_ul, col_sample = st.columns([2, 1])

    # ─────────────────────────────────────────────
    # SAMPLE CSV
    # ─────────────────────────────────────────────
    with col_sample:
        st.markdown("**Sample CSV format:**")

        sample = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003", "C004"],
            "total_views": [12, 85, 200, 5],
            "total_carts": [0, 4, 25, 0],
        })

        st.dataframe(sample, use_container_width=True, hide_index=True)

        csv_sample = sample.to_csv(index=False).encode()

        st.download_button(
            "⬇ Download sample CSV",
            data=csv_sample,
            file_name="sample_customers.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────
    # BATCH SCORING
    # ─────────────────────────────────────────────
    with col_ul:
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

        if uploaded:
            df_raw = pd.read_csv(uploaded)

            if (
                "total_views" not in df_raw.columns
                or "total_carts" not in df_raw.columns
            ):
                st.error(
                    "CSV must contain columns: `total_views` and `total_carts`."
                )

            else:
                with st.spinner("Scoring customers..."):
                    df_scored = score_dataframe(model, df_raw)

                st.session_state["df_scored"] = df_scored

                st.success(f"✅ Scored {len(df_scored):,} customers.")

                # ─────────────────────────────────────────────
                # KPI METRICS
                # ─────────────────────────────────────────────
                k1, k2, k3 = st.columns(3)

                with k1:
                    st.metric(
                        "Average Conversion Score",
                        round(df_scored["conversion_score"].mean(), 3)
                    )

                with k2:
                    st.metric(
                        "Average Structural Zero Prob.",
                        round(df_scored["p_zero"].mean(), 3)
                    )

                with k3:
                    st.metric(
                        "High Value Customers",
                        (
                            df_scored["segment"] == "High Value Customer"
                        ).sum()
                    )

                
                # ─────────────────────────────────────────────
                # DATA PREVIEW
                # ─────────────────────────────────────────────
                st.markdown("---")
                st.subheader("Scored Customer Preview")

                display_cols = [
                    c for c in df_scored.columns
                    if c in [
                        "customer_id",
                        "total_views",
                        "total_carts",
                        "p_zero",
                        "mu",
                        "conversion_score",
                        "segment"
                    ]
                ]

                st.dataframe(
                    df_scored[display_cols].round(4),
                    use_container_width=True,
                    hide_index=True,
                )
                # ─────────────────────────────────────────────
                # DOWNLOAD
                # ─────────────────────────────────────────────
                output_csv = df_scored.to_csv(index=False).encode()

                st.download_button(
                    "⬇ Download Scored CSV",
                    data=output_csv,
                    file_name="customers_scored.csv",
                    mime="text/csv",
                    type="primary",
                )
                
                # ─────────────────────────────────────────────
                # SEGMENT DISTRIBUTION
                # ─────────────────────────────────────────────
                st.markdown("## Customer Segment Distribution")

                seg_counts = df_scored["segment"].value_counts()
                seg_cols = st.columns(len(seg_counts))

                for i, (seg, cnt) in enumerate(seg_counts.items()):
                    cfg = SEGMENT_CONFIG.get(
                        seg,
                        {"color": "#adb5bd", "icon": "?"}
                    )
                    pct = cnt / len(df_scored) * 100

                    with seg_cols[i]:
                        st.markdown(
                            f"""
                            <div class="metric-card"
                                 style="text-align:center;
                                        border-top: 3px solid {cfg['color']};">
                                <div style="font-size:1.4rem">
                                    {cfg['icon']}
                                </div>
                                <div class="metric-label">{seg}</div>
                                <div class="metric-value"
                                     style="font-size:1.4rem">
                                    {cnt:,}
                                </div>
                                <div class="metric-sub">{pct:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.bar_chart(seg_counts)
                
                # ─────────────────────────────────────────────
                # TOP CUSTOMERS
                # ─────────────────────────────────────────────
                top_customers = df_scored.sort_values(
                    "conversion_score",
                    ascending=False
                ).head(10)

                st.subheader("Top 10 Priority Customers")

                st.dataframe(
                    top_customers[display_cols].round(4),
                    use_container_width=True,
                    hide_index=True
                )


                # ─────────────────────────────────────────────
                # ANALYTICS VISUALIZATIONS
                # ─────────────────────────────────────────────
                st.markdown("---")
                st.markdown("## 📊 Analytics Dashboard")

                # Score distribution
                st.subheader("Conversion Score Distribution")

                hist = np.histogram(
                    df_scored["conversion_score"],
                    bins=20
                )[0]

                hist_df = pd.DataFrame({"count": hist})
                st.bar_chart(hist_df)

                # Scatter plot
                st.subheader("Views vs Carts by Conversion Score")

                scatter_df = df_scored[
                    ["total_views", "total_carts", "conversion_score"]
                ]

                st.scatter_chart(
                    scatter_df,
                    x="total_views",
                    y="total_carts",
                    size="conversion_score"
                )
                

        else:
            st.info("Upload a CSV file above to begin batch scoring.")

# ═══════════════════════════════════════════════
# TAB 3: MODEL REFERENCE
# ═══════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">ZINB Model Structure</div>', unsafe_allow_html=True)

    st.markdown("""
    The **Zero-Inflated Negative Binomial (ZINB)** model addresses two data characteristics:
    - **Excess zeros**: Many customers never purchase (structural non-buyers)
    - **Overdispersion**: Variance of purchases exceeds the mean (α ≈ 0.67 > 0)

    #### Probability Mass Function
    """)

    st.latex(r"""
    P(Y_i = y) =
    \begin{cases}
        p_i + (1 - p_i) \cdot NB(0 \mid \mu_i, \alpha) & \text{if } y = 0 \\
        (1 - p_i) \cdot NB(y \mid \mu_i, \alpha) & \text{if } y > 0
    \end{cases}
    """)

    st.markdown("#### Linear Predictors")
    st.latex(r"""
    \text{logit}(p_i) = \gamma_0 + \gamma_1 \cdot \log\text{views}_i + \gamma_2 \cdot \log\text{carts}_i
    """)
    st.latex(r"""
    \log(\mu_i) = \beta_0 + \beta_1 \cdot \log\text{views}_i + \beta_2 \cdot \log\text{carts}_i
    """)

    st.markdown("#### Conversion Score Formula")
    st.latex(r"""
    \text{Score}_i = (1 - p_i) \times \mu_i
    """)

    st.markdown("""
    This score is the **expected number of purchases** for customer $i$, marginalizing over both
    the zero-inflation and count components. It serves as the statistically grounded lead score.
    """)

    st.markdown('<div class="section-header">Segmentation Rules</div>', unsafe_allow_html=True)

    seg_table = pd.DataFrame([
        {"Segment": "❄️ Cold User",           "P(Non-Buyer)": "> 0.85", "Score": "any",   "Action": "Low-cost retargeting"},
        {"Segment": "🔥 Warm Lead",            "P(Non-Buyer)": "0.50–0.85", "Score": "< 0.30", "Action": "Nudges & reminders"},
        {"Segment": "🎯 Potential Buyer",      "P(Non-Buyer)": "< 0.50", "Score": "0.30–1.00", "Action": "Discount offers"},
        {"Segment": "⭐ High Value Customer",  "P(Non-Buyer)": "< 0.50", "Score": "> 1.00", "Action": "Loyalty rewards"},
    ])
    st.dataframe(seg_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Project Information</div>', unsafe_allow_html=True)
    st.markdown("""
    | Field | Detail |
    |---|---|
    | **Student** | VISHNU B |
    | **Register No.** | 24MSKR0023 |
    | **Programme** | MSc Statistics (2024–26) |
    | **Guide** | ROHINI S NAIR |
    | **Submission** | March 2026 |
    | **Conference** | SCA-2026 |
    | **Dataset Size** | ~1.4 million users |
    | **Target Variable** | total_purchases |
    | **Selected Model** | Zero-Inflated Negative Binomial (ZINB) |
    | **Dispersion (α)** | ≈ 0.67 (confirms overdispersion) |
    """)

# ═══════════════════════════════════════════════
# TAB 4 : Diagbnostics Tab
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("## Model Diagnostics")

    if model_loaded:
        try:
            st.write("AIC:", model.aic)
            st.write("BIC:", model.bic)
            st.write("Log-Likelihood:", model.llf)
        except:
            st.warning("Diagnostics unavailable for this model object.")
    else:
        st.info("Diagnostics unavailable in demo mode.")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 ZINB Engine")
    st.markdown("---")
    st.markdown(f"**Model loaded:** {'✅ zinb_model.pkl' if model_loaded else '⚠️ Demo mode'}")
    st.markdown("**Dispersion α:** 0.67")
    st.markdown("**Dataset:** ~1.4M users")
    st.markdown("---")
    st.markdown("**Inputs:** `total_views`, `total_carts`")
    st.markdown("**Transform:** `log1p`")
    st.markdown("**Outputs:**")
    st.markdown("- P(non-buyer) = p")
    st.markdown("- E(purchases) = μ")
    st.markdown("- Score = (1−p)×μ")
    st.markdown("---")
    st.caption("MSc Data Science Project · VISHNU B\n24MSKR0023 · May 2026")

st.caption(
    "Built with Streamlit | ZINB Conversion Prediction Engine | MSc Data Science Project"
)