"""
conversion_engine.py
Customer Conversion Prediction Engine
Based on Zero-Inflated Negative Binomial (ZINB) Model

Project: Zero-Inflated Count Modeling of Customer Purchases in E-Commerce Platforms
Student: VISHNU B | Reg: 24MSKR0023 | MSc Data Science 2024-26
Guide: ROHINI S NAIR
"""

import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────
# 1. DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class ZINBOutputs:
    """Raw outputs extracted from the fitted ZINB model."""
    p_zero: float          # P(structural non-buyer)
    mu: float              # E(Y | buyer) — expected purchase count
    alpha: float           # Dispersion parameter


@dataclass
class CustomerProfile:
    """Full scoring and segmentation result for one customer."""
    total_views: int
    total_carts: int
    log_views: float
    log_carts: float
    p_zero: float
    mu: float
    alpha: float
    conversion_score: float
    segment: str
    recommendation: str
    segment_color: str     # for UI rendering


# ─────────────────────────────────────────────
# 2. MODEL LOADING
# ─────────────────────────────────────────────

def load_zinb_model(model_path: str = "zinb_model.pkl"):
    """
    Load the pre-trained ZINB model from disk.
    No retraining required.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# ─────────────────────────────────────────────
# 3. ZINB OUTPUT EXTRACTION
# ─────────────────────────────────────────────

def extract_zinb_outputs(model, log_views: float, log_carts: float) -> ZINBOutputs:
    """
    Extract structural zero probability (p_zero) and
    expected purchase intensity (mu) from the ZINB model.

    ZINB model has two components:
      - Zero component:  logit(p_i) = gamma_0 + gamma_1*log_views + gamma_2*log_carts
      - Count component: log(mu_i) = beta_0 + beta_1*log_views + beta_2*log_carts

    Parameters
    ----------
    model      : fitted ZINB model (statsmodels ZeroInflatedNegativeBinomialP)
    log_views  : log1p(total_views)
    log_carts  : log1p(total_carts)

    Returns
    -------
    ZINBOutputs with p_zero, mu, alpha
    """
    import pandas as pd

    # Build a single-row input dataframe
    X_new = np.array([[1.0, log_views, log_carts]])

    # statsmodels ZINB predict returns E[Y] by default (combined)
    # We need the component-level predictions
    # predict(which='mean')    → E[Y] = (1-p)*mu
    # predict(which='prob-zero') → P(Y=0) = p + (1-p)*NB(0|mu,alpha)
    # predict(which='lin')     → linear predictor for count component (log scale)
    # predict(which='prob-main') → p_i (zero-inflation probability)

    # Extract params by name
    params = model.params

    # Zero-inflation component: logit(p) = inflate_const + inflate_log_views*lv + inflate_log_carts*lc
    logit_p = (params["inflate_const"]
               + params["inflate_log_views"] * log_views
               + params["inflate_log_carts"] * log_carts)
    p_zero = float(1.0 / (1.0 + np.exp(-logit_p)))   # sigmoid

    # Count component: log(mu) = const + log_views*lv + log_carts*lc
    log_mu = (params["const"]
              + params["log_views"] * log_views
              + params["log_carts"] * log_carts)
    mu = float(np.exp(log_mu))

    # Dispersion
    alpha = float(params["alpha"])

    return ZINBOutputs(p_zero=p_zero, mu=mu, alpha=alpha)


# ─────────────────────────────────────────────
# 4. CONVERSION SCORE
# ─────────────────────────────────────────────

def compute_conversion_score(p_zero: float, mu: float) -> float:
    """
    Conversion Score Formula:

        Score_i = (1 - p_i) × μ_i

    Where:
        (1 - p_i) = probability of being a potential buyer
        μ_i       = expected purchase intensity

    Interpretation:
        High score → low non-buyer probability AND high expected purchases.
        This is the statistically grounded analog of a lead score.

    Score is NOT normalized here; it is on the scale of expected purchases
    adjusted for zero-inflation. Normalization (0–100) is done in the UI layer.
    """
    score = (1.0 - p_zero) * mu
    return round(score, 6)


# ─────────────────────────────────────────────
# 5. CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────

# Segment thresholds — tune these based on your dataset's score distribution
# Recommended: use quantiles of score from training set for production

SEGMENT_RULES = [
    # (max_p_zero, min_score, segment_name, color)
    (0.85, 0.00, "Cold User",          "#6c757d"),   # high p_zero, any score
    (1.00, 0.00, "Warm Lead",          "#ffc107"),   # moderate p_zero, low score
    (1.00, 0.30, "Potential Buyer",    "#17a2b8"),   # low p_zero, moderate score
    (1.00, 1.00, "High Value Customer","#28a745"),   # low p_zero, high score
]

RECOMMENDATIONS = {
    "Cold User": (
        "Low-cost retargeting campaigns. "
        "Show browse-based ads to re-engage. "
        "Do not invest heavily — conversion probability is low."
    ),
    "Warm Lead": (
        "Send gentle nudges and reminders. "
        "Highlight items left in wishlist or previously viewed. "
        "Email with social proof (reviews, popularity)."
    ),
    "Potential Buyer": (
        "Offer targeted discounts or limited-time deals. "
        "Send cart abandonment emails if applicable. "
        "Free shipping threshold nudges work well for this segment."
    ),
    "High Value Customer": (
        "Activate loyalty rewards and premium offers. "
        "Early access to new products or flash sales. "
        "Personalized cross-sell and upsell recommendations."
    ),
}


def assign_segment(p_zero: float, score: float) -> tuple[str, str]:
    """
    Assign customer segment based on p_zero and conversion score.

    Segmentation logic:
    ┌──────────────────────┬──────────────┬──────────────┐
    │ Segment              │ p_zero       │ Score        │
    ├──────────────────────┼──────────────┼──────────────┤
    │ Cold User            │ > 0.85       │ any          │
    │ Warm Lead            │ 0.50–0.85    │ < 0.30       │
    │ Potential Buyer      │ < 0.50       │ 0.30–1.00    │
    │ High Value Customer  │ < 0.50       │ > 1.00       │
    └──────────────────────┴──────────────┴──────────────┘

    Returns (segment_name, hex_color)
    """
    if p_zero > 0.85:
        return "Cold User", "#6c757d"
    elif p_zero > 0.50:
        return "Warm Lead", "#ffc107"
    elif score >= 1.00:
        return "High Value Customer", "#28a745"
    else:
        return "Potential Buyer", "#17a2b8"


# ─────────────────────────────────────────────
# 6. MAIN ENGINE FUNCTION
# ─────────────────────────────────────────────

def predict_customer(
    model,
    total_views: int,
    total_carts: int,
) -> CustomerProfile:
    """
    Full pipeline: raw inputs → ZINB outputs → score → segment → recommendation.

    Parameters
    ----------
    model        : loaded ZINB model
    total_views  : raw total page views for the customer
    total_carts  : raw total cart additions for the customer

    Returns
    -------
    CustomerProfile with all computed fields
    """
    # Step 1: Feature engineering (matching training pipeline exactly)
    log_views = float(np.log1p(total_views))
    log_carts = float(np.log1p(total_carts))

    # Step 2: Extract ZINB model outputs
    zinb_out = extract_zinb_outputs(model, log_views, log_carts)

    # Step 3: Compute conversion score
    score = compute_conversion_score(zinb_out.p_zero, zinb_out.mu)

    # Step 4: Assign segment
    segment, color = assign_segment(zinb_out.p_zero, score)

    # Step 5: Fetch recommendation
    recommendation = RECOMMENDATIONS[segment]

    return CustomerProfile(
        total_views=total_views,
        total_carts=total_carts,
        log_views=log_views,
        log_carts=log_carts,
        p_zero=zinb_out.p_zero,
        mu=zinb_out.mu,
        alpha=zinb_out.alpha,
        conversion_score=score,
        segment=segment,
        recommendation=recommendation,
        segment_color=color,
    )


# ─────────────────────────────────────────────
# 7. BATCH SCORING (for full dataset)
# ─────────────────────────────────────────────

def score_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score an entire DataFrame of customers.

    Input df must have columns: total_views, total_carts

    Returns df with appended columns:
        log_views, log_carts, p_zero, mu, alpha,
        conversion_score, segment
    """
    df = df.copy()
    df["log_views"] = np.log1p(df["total_views"])
    df["log_carts"] = np.log1p(df["total_carts"])

    df["const"] = 1.0
    X = df[["const", "log_views", "log_carts"]]

    params = model.params

    logit_p = (params["inflate_const"]
               + params["inflate_log_views"] * df["log_views"]
               + params["inflate_log_carts"] * df["log_carts"])
    df["p_zero"] = 1.0 / (1.0 + np.exp(-logit_p))

    log_mu = (params["const"]
              + params["log_views"] * df["log_views"]
              + params["log_carts"] * df["log_carts"])
    df["mu"] = np.exp(log_mu)

    df["alpha"] = float(params["alpha"])
    df["conversion_score"] = (1 - df["p_zero"]) * df["mu"]

    df["segment"] = df.apply(
        lambda row: assign_segment(row["p_zero"], row["conversion_score"])[0],
        axis=1,
    )

    return df


# ─────────────────────────────────────────────
# 8. QUICK TEST (run directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate ZINB outputs for testing without the actual model
    class MockModel:
        """Mimics a fitted ZINB model for development/testing."""
        params = {"alpha": 0.67}

        def predict(self, X, which="mean"):
            log_views = X["log_views"].values[0]
            log_carts = X["log_carts"].values[0]
            if which == "prob-main":
                # Simulate: more carts → less likely non-buyer
                return np.array([max(0.05, 0.95 - 0.15 * log_carts)])
            elif which == "lin":
                # Simulate: log(mu) increases with carts
                return np.array([0.3 + 0.4 * log_carts + 0.1 * log_views])

    mock_model = MockModel()

    test_cases = [
        (5,   0,  "Casual browser, no carts"),
        (50,  2,  "Active browser, few carts"),
        (120, 10, "Engaged shopper"),
        (300, 40, "High-intent power user"),
    ]

    print("=" * 70)
    print("CUSTOMER CONVERSION PREDICTION ENGINE — TEST RUN")
    print("=" * 70)

    for views, carts, label in test_cases:
        profile = predict_customer(mock_model, views, carts)
        print(f"\n[{label}]")
        print(f"  Views: {views:>4}  |  Carts: {carts:>3}")
        print(f"  log_views = {profile.log_views:.4f}  |  log_carts = {profile.log_carts:.4f}")
        print(f"  P(non-buyer) = {profile.p_zero:.4f}")
        print(f"  E(purchases | buyer) = {profile.mu:.4f}")
        print(f"  Conversion Score = {profile.conversion_score:.4f}")
        print(f"  Segment: {profile.segment}")
        print(f"  Recommendation: {profile.recommendation}")

    print("\n" + "=" * 70)