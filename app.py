import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(
    page_title="Food Delivery Statistical Inference",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Food Delivery Time & Customer Ratings")
st.caption("Statistical Inference Lab Project | Exploratory analysis, confidence intervals, hypothesis tests, and association analysis")

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "food_order.csv")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        if not os.path.exists(DEFAULT_DATA):
            return None
        raw = pd.read_csv(DEFAULT_DATA)

    expected = {
        "delivery_time", "food_preparation_time", "rating",
        "day_of_the_week", "cuisine_type", "cost_of_the_order"
    }
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(sorted(missing)))

    df = raw.copy()
    df["delivery_time"] = pd.to_numeric(df["delivery_time"], errors="coerce")
    df["food_preparation_time"] = pd.to_numeric(df["food_preparation_time"], errors="coerce")
    df["cost_of_the_order"] = pd.to_numeric(df["cost_of_the_order"], errors="coerce")
    df["rating_numeric"] = pd.to_numeric(df["rating"].replace("Not given", np.nan), errors="coerce")
    df["day_of_the_week"] = df["day_of_the_week"].astype(str).str.strip().str.title()
    df = df.dropna(subset=["delivery_time", "food_preparation_time", "cost_of_the_order"])
    return df

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload a replacement CSV (optional)", type=["csv"])
    st.markdown("---")
    st.header("Analysis settings")
    benchmark = st.number_input("Delivery-time benchmark (minutes)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    alpha = st.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05)
    st.caption("The benchmark and significance level are configurable.")

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if df is None:
    st.error("Dataset not found. Place food_order.csv beside app.py or upload a CSV using the sidebar.")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} orders")

# Summary metrics
rating_df = df.dropna(subset=["rating_numeric"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total orders", f"{len(df):,}")
c2.metric("Mean delivery time", f"{df['delivery_time'].mean():.2f} min")
c3.metric("Rated orders", f"{len(rating_df):,}")
c4.metric("Mean recorded rating", f"{rating_df['rating_numeric'].mean():.2f} / 5" if len(rating_df) else "N/A")

st.info("Ratings marked 'Not given' are treated as missing, not as zero or a low rating. Rating-based analyses use only recorded numeric ratings.")

tabs = st.tabs([
    "Overview & EDA",
    "Confidence Interval",
    "Hypothesis Tests",
    "Correlation",
    "Chi-Square",
    "Data Table",
])

with tabs[0]:
    st.subheader("Dataset overview")
    left, right = st.columns(2)
    with left:
        st.markdown("**Delivery-time distribution**")
        fig = px.histogram(df, x="delivery_time", nbins=20, marginal="box",
                           labels={"delivery_time": "Delivery time (minutes)"},
                           title="Distribution of delivery times")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Customer rating distribution**")
        if len(rating_df):
            rating_counts = rating_df["rating_numeric"].value_counts().sort_index().rename_axis("Rating").reset_index(name="Orders")
            fig = px.bar(rating_counts, x="Rating", y="Orders", title="Recorded customer ratings",
                         labels={"Rating": "Customer rating", "Orders": "Number of orders"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric ratings found.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Delivery time by day**")
        fig = px.box(df, x="day_of_the_week", y="delivery_time", points="outliers",
                     labels={"day_of_the_week": "Day", "delivery_time": "Delivery time (minutes)"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Food preparation vs delivery time**")
        fig = px.scatter(df, x="food_preparation_time", y="delivery_time", color="day_of_the_week",
                         labels={"food_preparation_time": "Food preparation time (minutes)",
                                 "delivery_time": "Delivery time (minutes)"})
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Numeric summary**")
    st.dataframe(df[["delivery_time", "food_preparation_time", "cost_of_the_order"]].describe().T, use_container_width=True)

with tabs[1]:
    st.subheader("Confidence interval for mean delivery time")
    values = df["delivery_time"].dropna().astype(float)
    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1)
    if n >= 2:
        tcrit = stats.t.ppf(0.975, df=n-1)
        margin = tcrit * sd / np.sqrt(n)
        lower, upper = mean - margin, mean + margin
        a, b, c = st.columns(3)
        a.metric("Sample mean", f"{mean:.2f} min")
        b.metric("95% CI lower bound", f"{lower:.2f} min")
        c.metric("95% CI upper bound", f"{upper:.2f} min")
        st.write(f"Using a t-interval: **{mean:.2f} ± {margin:.2f} minutes**.")
        st.markdown("**Interpretation:** Under the usual sampling assumptions, this interval gives a plausible range for the population mean delivery time.")
    else:
        st.warning("At least two valid delivery-time values are required.")

    st.subheader("Confidence interval for mean recorded customer rating")
    r = rating_df["rating_numeric"].dropna().astype(float)
    if len(r) >= 2:
        rmean, rsd, rn = r.mean(), r.std(ddof=1), len(r)
        rmargin = stats.t.ppf(0.975, rn-1) * rsd / np.sqrt(rn)
        st.write(f"Recorded ratings: **{rn:,}**")
        st.write(f"Mean rating: **{rmean:.3f} / 5**")
        st.write(f"95% confidence interval: **({rmean-rmargin:.3f}, {rmean+rmargin:.3f})**")
    else:
        st.warning("Not enough numeric ratings for a confidence interval.")

with tabs[2]:
    st.subheader("Test 1 — One-sample t-test")
    st.markdown(f"H₀: population mean delivery time = **{benchmark:.1f} minutes**  \nH₁: population mean delivery time differs from **{benchmark:.1f} minutes**.")
    tstat, pvalue = stats.ttest_1samp(df["delivery_time"].dropna(), popmean=benchmark)
    st.write(f"t-statistic: **{tstat:.4f}**")
    st.write(f"p-value: **{pvalue:.6g}**")
    st.write(f"Decision at α = {alpha}: **{'Reject H₀' if pvalue < alpha else 'Fail to reject H₀'}**")
    st.caption("A non-significant result does not prove H₀ is true; it means there is insufficient evidence against it at the chosen significance level.")

    st.divider()
    st.subheader("Test 2 — Welch two-sample t-test (weekday vs weekend)")
    groups = {name: grp["delivery_time"].dropna().astype(float) for name, grp in df.groupby("day_of_the_week")}
    weekday = groups.get("Weekday", pd.Series(dtype=float))
    weekend = groups.get("Weekend", pd.Series(dtype=float))
    if len(weekday) >= 2 and len(weekend) >= 2:
        t2, p2 = stats.ttest_ind(weekday, weekend, equal_var=False)
        a, b = st.columns(2)
        a.metric("Weekday mean", f"{weekday.mean():.2f} min", f"n={len(weekday)}")
        b.metric("Weekend mean", f"{weekend.mean():.2f} min", f"n={len(weekend)}")
        st.write(f"Welch t-statistic: **{t2:.4f}**")
        st.write(f"p-value: **{p2:.6g}**")
        st.write(f"Decision at α = {alpha}: **{'Reject H₀' if p2 < alpha else 'Fail to reject H₀'}**")
        st.markdown("H₀: weekday and weekend mean delivery times are equal. H₁: their means differ.")
    else:
        st.warning("Need at least two observations in both weekday and weekend groups.")

with tabs[3]:
    st.subheader("Correlation between delivery time and customer rating")
    corr_data = df.dropna(subset=["rating_numeric", "delivery_time"])
    if len(corr_data) >= 3 and corr_data["rating_numeric"].nunique() > 1:
        pearson_r, pearson_p = stats.pearsonr(corr_data["delivery_time"], corr_data["rating_numeric"])
        spearman_r, spearman_p = stats.spearmanr(corr_data["delivery_time"], corr_data["rating_numeric"])
        a, b = st.columns(2)
        a.metric("Pearson correlation (r)", f"{pearson_r:.3f}")
        b.metric("Spearman correlation (ρ)", f"{spearman_r:.3f}")
        st.write(f"Pearson p-value: **{pearson_p:.6g}**")
        st.write(f"Spearman p-value: **{spearman_p:.6g}**")
        st.markdown("**Interpretation guide:** values near +1 indicate positive association, values near −1 indicate negative association, and values near 0 indicate little linear/monotonic association. Correlation does not establish causation.")
        fig = px.strip(corr_data, x="rating_numeric", y="delivery_time", points="all",
                       labels={"rating_numeric": "Customer rating", "delivery_time": "Delivery time (minutes)"},
                       title="Delivery time for each recorded customer rating")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough usable ratings for correlation analysis.")

    st.subheader("Correlation matrix")
    corr_cols = ["delivery_time", "food_preparation_time", "cost_of_the_order"]
    if len(rating_df):
        corr_frame = df[corr_cols + ["rating_numeric"]].corr(numeric_only=True)
    else:
        corr_frame = df[corr_cols].corr(numeric_only=True)
    fig = px.imshow(corr_frame, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Pearson correlation matrix")
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Chi-square test of independence")
    st.markdown("This test checks whether recorded rating categories are associated with the day of the week. It uses only orders with recorded ratings.")
    chi_data = df.dropna(subset=["rating_numeric", "day_of_the_week"]).copy()
    if len(chi_data) and chi_data["rating_numeric"].nunique() > 1 and chi_data["day_of_the_week"].nunique() > 1:
        chi_data["rating_category"] = chi_data["rating_numeric"].astype(int).astype(str)
        table = pd.crosstab(chi_data["day_of_the_week"], chi_data["rating_category"])
        chi2, pchi, dof, expected = stats.chi2_contingency(table)
        st.markdown("H₀: rating category and day of week are independent.  \nH₁: they are associated.")
        st.dataframe(table, use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Chi-square statistic", f"{chi2:.3f}")
        b.metric("Degrees of freedom", str(dof))
        c.metric("p-value", f"{pchi:.6g}")
        st.write(f"Decision at α = {alpha}: **{'Reject H₀' if pchi < alpha else 'Fail to reject H₀'}**")
        if (expected < 5).any():
            st.warning("Some expected cell counts are below 5; interpret the chi-square approximation cautiously.")
        fig = px.bar(table.reset_index().melt(id_vars="day_of_the_week", var_name="Rating", value_name="Orders"),
                     x="Rating", y="Orders", color="day_of_the_week", barmode="group",
                     labels={"day_of_the_week": "Day of week"}, title="Recorded ratings by day of week")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough categorical variation to perform this test.")

with tabs[5]:
    st.subheader("Data preview")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df, use_container_width=True, height=450)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned dataset (CSV)", csv_bytes, "food_order_cleaned.csv", "text/csv")
    st.markdown("**Column notes**")
    st.markdown("""
    - `delivery_time`: delivery duration in minutes.
    - `food_preparation_time`: time taken to prepare the order, in minutes.
    - `rating`: original customer rating; `Not given` values are treated as missing.
    - `rating_numeric`: numeric version used for rating-based analyses.
    - `day_of_the_week`: weekday/weekend category.
    """)

st.divider()
st.caption("Statistical Inference Lab | Results are computed from the loaded dataset. Interpret p-values alongside effect sizes, assumptions, and data limitations.")
