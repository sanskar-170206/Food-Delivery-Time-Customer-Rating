import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Food Delivery Statistical Inference",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Food Delivery Time & Customer Ratings")
st.caption(
    "Statistical Inference Lab | EDA, Confidence Intervals, "
    "Hypothesis Testing, Correlation, Chi-Square and ML Prediction"
)

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "food_order.csv")


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    elif os.path.exists(DEFAULT_DATA):
        raw = pd.read_csv(DEFAULT_DATA)
    else:
        return None

    required = [
        "delivery_time",
        "food_preparation_time",
        "rating",
        "day_of_the_week",
        "cuisine_type",
        "cost_of_the_order"
    ]

    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    df = raw.copy()

    for col in [
        "delivery_time",
        "food_preparation_time",
        "cost_of_the_order"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rating_numeric"] = pd.to_numeric(
        df["rating"].replace("Not given", np.nan),
        errors="coerce"
    )

    df["day_of_the_week"] = (
        df["day_of_the_week"].astype(str).str.strip().str.title()
    )
    df["cuisine_type"] = df["cuisine_type"].astype(str).str.strip()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[
        "delivery_time",
        "food_preparation_time",
        "cost_of_the_order"
    ])

    return df


def make_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ])


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"]
    )

    st.divider()
    st.header("Analysis Settings")

    benchmark = st.number_input(
        "Delivery-time benchmark (minutes)",
        min_value=1.0,
        max_value=300.0,
        value=30.0,
        step=1.0
    )

    alpha = st.select_slider(
        "Significance level (α)",
        options=[0.01, 0.05, 0.10],
        value=0.05
    )

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if df is None:
    st.error(
        "Dataset not found. Place food_order.csv beside app.py "
        "or upload your CSV in the sidebar."
    )
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} orders")

rating_df = df.dropna(subset=["rating_numeric"])
rating_df = rating_df[rating_df["rating_numeric"].between(1, 5)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Orders", f"{len(df):,}")
c2.metric("Mean Delivery Time", f"{df['delivery_time'].mean():.2f} min")
c3.metric("Rated Orders", f"{len(rating_df):,}")
c4.metric(
    "Mean Customer Rating",
    f"{rating_df['rating_numeric'].mean():.2f}/5"
    if len(rating_df) else "N/A"
)

st.info("Ratings marked 'Not given' are treated as missing, not zero.")

tabs = st.tabs([
    "Overview & EDA",
    "Confidence Interval",
    "Hypothesis Tests",
    "Correlation",
    "Chi-Square",
    "Data Table",
    "Prediction"
])

# TAB 1: OVERVIEW AND EDA

with tabs[0]:
    st.subheader("Dataset Overview")

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df,
            x="delivery_time",
            nbins=20,
            marginal="box",
            title="Distribution of Delivery Times",
            labels={"delivery_time": "Delivery Time (minutes)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if len(rating_df):
            rating_counts = (
                rating_df["rating_numeric"]
                .value_counts()
                .sort_index()
                .rename_axis("Rating")
                .reset_index(name="Orders")
            )

            fig = px.bar(
                rating_counts,
                x="Rating",
                y="Orders",
                title="Customer Rating Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric customer ratings found.")

    left, right = st.columns(2)

    with left:
        fig = px.box(
            df,
            x="day_of_the_week",
            y="delivery_time",
            points="outliers",
            title="Delivery Time by Day"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            df,
            x="food_preparation_time",
            y="delivery_time",
            color="day_of_the_week",
            title="Food Preparation vs Delivery Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric Summary")
    st.dataframe(
        df[[
            "delivery_time",
            "food_preparation_time",
            "cost_of_the_order"
        ]].describe().T,
        use_container_width=True
    )


# TAB 2: CONFIDENCE INTERVALS

with tabs[1]:
    st.subheader("Confidence Interval for Mean Delivery Time")

    values = df["delivery_time"].dropna()
    n = len(values)

    if n >= 2:
        mean = values.mean()
        sd = values.std(ddof=1)
        tcrit = stats.t.ppf(0.975, n - 1)
        margin = tcrit * sd / np.sqrt(n)

        a, b, c = st.columns(3)
        a.metric("Sample Mean", f"{mean:.2f} min")
        b.metric("95% CI Lower Bound", f"{mean-margin:.2f} min")
        c.metric("95% CI Upper Bound", f"{mean+margin:.2f} min")

        st.write(
            f"95% confidence interval: "
            f"({mean-margin:.2f}, {mean+margin:.2f}) minutes"
        )
    else:
        st.warning("At least two observations are required.")

    st.divider()
    st.subheader("Confidence Interval for Customer Rating")

    ratings = rating_df["rating_numeric"]

    if len(ratings) >= 2:
        rmean = ratings.mean()
        rsd = ratings.std(ddof=1)
        rn = len(ratings)
        rmargin = stats.t.ppf(0.975, rn - 1) * rsd / np.sqrt(rn)

        st.write(f"Number of recorded ratings: {rn}")
        st.write(f"Mean rating: {rmean:.3f}/5")
        st.write(
            f"95% confidence interval: "
            f"({rmean-rmargin:.3f}, {rmean+rmargin:.3f})"
        )
    else:
        st.warning("Not enough numeric ratings.")


# TAB 3: HYPOTHESIS TESTS

with tabs[2]:
    st.subheader("One-Sample t-Test")

    delivery_values = df["delivery_time"].dropna()

    st.write(f"H₀: Mean delivery time = {benchmark:.1f} minutes")
    st.write(f"H₁: Mean delivery time differs from {benchmark:.1f} minutes")

    if len(delivery_values) >= 2:
        tstat, pvalue = stats.ttest_1samp(
            delivery_values,
            popmean=benchmark
        )

        st.write(f"t-statistic: {tstat:.4f}")
        st.write(f"p-value: {pvalue:.6g}")
        st.write(
            f"Decision: "
            f"{'Reject H₀' if pvalue < alpha else 'Fail to reject H₀'}"
        )

    st.divider()
    st.subheader("Welch Two-Sample t-Test: Weekday vs Weekend")

    groups = {
        name: grp["delivery_time"].dropna()
        for name, grp in df.groupby("day_of_the_week")
    }

    weekday = groups.get("Weekday", pd.Series(dtype=float))
    weekend = groups.get("Weekend", pd.Series(dtype=float))

    if len(weekday) >= 2 and len(weekend) >= 2:
        t2, p2 = stats.ttest_ind(
            weekday,
            weekend,
            equal_var=False
        )

        a, b = st.columns(2)
        a.metric("Weekday Mean", f"{weekday.mean():.2f} min")
        b.metric("Weekend Mean", f"{weekend.mean():.2f} min")

        st.write(f"t-statistic: {t2:.4f}")
        st.write(f"p-value: {p2:.6g}")
        st.write(
            f"Decision: "
            f"{'Reject H₀' if p2 < alpha else 'Fail to reject H₀'}"
        )
    else:
        st.warning(
            "Need at least two observations in both weekday and weekend groups."
        )


# TAB 4: CORRELATION

with tabs[3]:
    st.subheader("Delivery Time and Customer Rating Correlation")

    corr_data = df.dropna(
        subset=["rating_numeric", "delivery_time"]
    )

    if (
        len(corr_data) >= 3
        and corr_data["rating_numeric"].nunique() > 1
        and corr_data["delivery_time"].nunique() > 1
    ):
        pearson_r, pearson_p = stats.pearsonr(
            corr_data["delivery_time"],
            corr_data["rating_numeric"]
        )

        spearman_r, spearman_p = stats.spearmanr(
            corr_data["delivery_time"],
            corr_data["rating_numeric"]
        )

        a, b = st.columns(2)
        a.metric("Pearson Correlation", f"{pearson_r:.3f}")
        b.metric("Spearman Correlation", f"{spearman_r:.3f}")

        st.write(f"Pearson p-value: {pearson_p:.6g}")
        st.write(f"Spearman p-value: {spearman_p:.6g}")

        fig = px.strip(
            corr_data,
            x="rating_numeric",
            y="delivery_time",
            title="Delivery Time by Customer Rating"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough usable ratings for correlation analysis.")

    st.subheader("Correlation Matrix")

    corr_cols = [
        "delivery_time",
        "food_preparation_time",
        "cost_of_the_order",
        "rating_numeric"
    ]

    corr_frame = df[corr_cols].corr(numeric_only=True)

    fig = px.imshow(
        corr_frame,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)


# TAB 5: CHI-SQUARE TEST

with tabs[4]:
    st.subheader("Chi-Square Test of Independence")

    chi_data = df.dropna(
        subset=["rating_numeric", "day_of_the_week"]
    ).copy()

    chi_data = chi_data[
        chi_data["rating_numeric"].between(1, 5)
    ]

    if (
        len(chi_data)
        and chi_data["rating_numeric"].nunique() > 1
        and chi_data["day_of_the_week"].nunique() > 1
    ):
        chi_data["rating_category"] = (
            chi_data["rating_numeric"].astype(int).astype(str)
        )

        table = pd.crosstab(
            chi_data["day_of_the_week"],
            chi_data["rating_category"]
        )

        chi2, pchi, dof, expected = stats.chi2_contingency(table)

        st.write("H₀: Rating category and day of week are independent.")
        st.write("H₁: Rating category and day of week are associated.")

        st.dataframe(table, use_container_width=True)

        a, b, c = st.columns(3)
        a.metric("Chi-Square Statistic", f"{chi2:.3f}")
        b.metric("Degrees of Freedom", str(dof))
        c.metric("p-value", f"{pchi:.6g}")

        st.write(
            f"Decision: "
            f"{'Reject H₀' if pchi < alpha else 'Fail to reject H₀'}"
        )

        if (expected < 5).any():
            st.warning(
                "Some expected cell counts are below 5. Interpret cautiously."
            )

        fig = px.bar(
            table.reset_index().melt(
                id_vars="day_of_the_week",
                var_name="Rating",
                value_name="Orders"
            ),
            x="Rating",
            y="Orders",
            color="day_of_the_week",
            barmode="group",
            title="Recorded Ratings by Day"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough categorical variation for this test.")


# TAB 6: DATA TABLE

with tabs[5]:
    st.subheader("Data Preview")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df, use_container_width=True, height=450)

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Cleaned Dataset",
        csv_bytes,
        "food_order_cleaned.csv",
        "text/csv"
    )

    st.markdown("""
    **Column Notes**

    - `delivery_time`: Delivery duration in minutes.
    - `food_preparation_time`: Food preparation time in minutes.
    - `rating`: Original customer rating.
    - `rating_numeric`: Numeric customer rating.
    - `day_of_the_week`: Weekday/weekend category.
    - `cuisine_type`: Cuisine category.
    - `cost_of_the_order`: Order cost.
    """)


# TAB 7: MACHINE LEARNING PREDICTION

with tabs[6]:
    st.subheader("Food Delivery Time and Customer Rating Prediction")

    st.write(
        "Random Forest regression models estimate delivery time "
        "and customer ratings using historical orders."
    )

    delivery_features = [
        "food_preparation_time",
        "cost_of_the_order",
        "cuisine_type",
        "day_of_the_week"
    ]

    delivery_data = df[
        delivery_features + ["delivery_time"]
    ].copy()

    delivery_data = delivery_data.replace(
        [np.inf, -np.inf], np.nan
    )

    delivery_data = delivery_data.dropna(
        subset=delivery_features + ["delivery_time"]
    )

    delivery_data = delivery_data[
        (delivery_data["delivery_time"] > 0)
        & (delivery_data["food_preparation_time"] > 0)
        & (delivery_data["cost_of_the_order"] > 0)
    ]

    delivery_model = None
    rating_model = None

    if len(delivery_data) < 10:
        st.warning(
            "At least 10 valid orders are required for delivery-time prediction."
        )
    else:
        # DELIVERY-TIME MODEL

        X_delivery = delivery_data[delivery_features]
        y_delivery = delivery_data["delivery_time"]

        delivery_model = Pipeline([
            (
                "preprocessing",
                make_preprocessor(
                    ["food_preparation_time", "cost_of_the_order"],
                    ["cuisine_type", "day_of_the_week"]
                )
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2
                )
            )
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X_delivery,
            y_delivery,
            test_size=0.2,
            random_state=42
        )

        delivery_model.fit(X_train, y_train)
        delivery_predictions = delivery_model.predict(X_test)

        delivery_mae = mean_absolute_error(
            y_test, delivery_predictions
        )
        delivery_rmse = np.sqrt(
            mean_squared_error(y_test, delivery_predictions)
        )
        delivery_r2 = (
            r2_score(y_test, delivery_predictions)
            if len(y_test) >= 2 else np.nan
        )

        st.markdown("### Delivery-Time Model Performance")

        a, b, c = st.columns(3)
        a.metric("MAE", f"{delivery_mae:.2f} minutes")
        b.metric("RMSE", f"{delivery_rmse:.2f} minutes")
        c.metric(
            "R² Score",
            f"{delivery_r2:.3f}"
            if np.isfinite(delivery_r2) else "N/A"
        )

        st.caption(
            "Lower MAE and RMSE are better. R² can be negative."
        )

        delivery_comparison = pd.DataFrame({
            "Actual Delivery Time": y_test.to_numpy(),
            "Predicted Delivery Time": delivery_predictions
        })

        fig = px.scatter(
            delivery_comparison,
            x="Actual Delivery Time",
            y="Predicted Delivery Time",
            title="Actual vs Predicted Delivery Times"
        )

        low = min(
            delivery_comparison["Actual Delivery Time"].min(),
            delivery_comparison["Predicted Delivery Time"].min()
        )
        high = max(
            delivery_comparison["Actual Delivery Time"].max(),
            delivery_comparison["Predicted Delivery Time"].max()
        )

        fig.add_shape(
            type="line",
            x0=low,
            y0=low,
            x1=high,
            y1=high,
            line=dict(dash="dash")
        )

        st.plotly_chart(fig, use_container_width=True)

        # CUSTOMER RATING MODEL

        st.divider()
        st.markdown("### Customer Rating Prediction")

        rating_features = [
            "food_preparation_time",
            "delivery_time",
            "cost_of_the_order",
            "cuisine_type",
            "day_of_the_week"
        ]

        rating_data = df[
            rating_features + ["rating_numeric"]
        ].copy()

        rating_data = rating_data.replace(
            [np.inf, -np.inf], np.nan
        )

        rating_data = rating_data.dropna(
            subset=rating_features + ["rating_numeric"]
        )

        rating_data = rating_data[
            rating_data["rating_numeric"].between(1, 5)
            & (rating_data["delivery_time"] > 0)
            & (rating_data["food_preparation_time"] > 0)
            & (rating_data["cost_of_the_order"] > 0)
        ]

        if len(rating_data) < 10:
            st.warning(
                f"Only {len(rating_data)} usable rated orders are available. "
                "At least 10 are required to train the rating model."
            )
        else:
            rating_model = Pipeline([
                (
                    "preprocessing",
                    make_preprocessor(
                        [
                            "food_preparation_time",
                            "delivery_time",
                            "cost_of_the_order"
                        ],
                        ["cuisine_type", "day_of_the_week"]
                    )
                ),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        min_samples_leaf=2
                    )
                )
            ])

            X_rating = rating_data[rating_features]
            y_rating = rating_data["rating_numeric"]

            Xr_train, Xr_test, yr_train, yr_test = train_test_split(
                X_rating,
                y_rating,
                test_size=0.2,
                random_state=42
            )

            rating_model.fit(Xr_train, yr_train)

            rating_predictions = np.clip(
                rating_model.predict(Xr_test), 1, 5
            )

            rating_mae = mean_absolute_error(
                yr_test, rating_predictions
            )
            rating_rmse = np.sqrt(
                mean_squared_error(yr_test, rating_predictions)
            )
            rating_r2 = (
                r2_score(yr_test, rating_predictions)
                if len(yr_test) >= 2 else np.nan
            )

            st.write(
                f"Orders with recorded ratings: {len(rating_data):,}"
            )

            a, b, c = st.columns(3)
            a.metric("Rating MAE", f"{rating_mae:.3f} stars")
            b.metric("Rating RMSE", f"{rating_rmse:.3f} stars")
            c.metric(
                "Rating R² Score",
                f"{rating_r2:.3f}"
                if np.isfinite(rating_r2) else "N/A"
            )

            st.caption(
                "These metrics evaluate the rating model on held-out data."
            )

            rating_comparison = pd.DataFrame({
                "Actual Rating": yr_test.to_numpy(),
                "Predicted Rating": rating_predictions
            })

            fig = px.scatter(
                rating_comparison,
                x="Actual Rating",
                y="Predicted Rating",
                title="Actual vs Predicted Customer Ratings",
                range_x=[1, 5],
                range_y=[1, 5]
            )

            fig.add_shape(
                type="line",
                x0=1,
                y0=1,
                x1=5,
                y1=5,
                line=dict(dash="dash")
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                rating_comparison.round(2),
                use_container_width=True
            )

        # NEW ORDER PREDICTION FORM

        st.divider()
        st.markdown("### Predict a New Order")

        cuisines = sorted(
            delivery_data["cuisine_type"].astype(str).unique().tolist()
        )
        days = sorted(
            delivery_data["day_of_the_week"].astype(str).unique().tolist()
        )

        with st.form("prediction_form"):
            cuisine = st.selectbox("Cuisine Type", cuisines)

            prep_time = st.number_input(
                "Food Preparation Time (minutes)",
                min_value=1,
                max_value=180,
                value=25
            )

            cost = st.number_input(
                "Order Cost",
                min_value=0.01,
                value=15.0,
                step=1.0
            )

            day = st.selectbox("Day of the Week", days)

            submitted = st.form_submit_button(
                "Predict Delivery Time and Customer Rating"
            )

        if submitted and delivery_model is not None:
            new_order = pd.DataFrame([{
                "food_preparation_time": prep_time,
                "cost_of_the_order": cost,
                "cuisine_type": cuisine,
                "day_of_the_week": day
            }])

            predicted_time = float(
                delivery_model.predict(new_order)[0]
            )
            predicted_time = max(0.0, predicted_time)

            a, b = st.columns(2)

            a.metric(
                "Estimated Delivery Time",
                f"{predicted_time:.1f} minutes"
            )

            if rating_model is not None:
                new_rating_order = pd.DataFrame([{
                    "food_preparation_time": prep_time,
                    "delivery_time": predicted_time,
                    "cost_of_the_order": cost,
                    "cuisine_type": cuisine,
                    "day_of_the_week": day
                }])

                predicted_rating = float(
                    np.clip(
                        rating_model.predict(new_rating_order)[0],
                        1,
                        5
                    )
                )

                b.metric(
                    "Predicted Customer Rating",
                    f"{predicted_rating:.2f}/5 ⭐"
                )

                st.success(
                    f"Estimated delivery time: {predicted_time:.1f} minutes\n\n"
                    f"Predicted customer rating: {predicted_rating:.2f}/5"
                )
            else:
                b.metric("Predicted Customer Rating", "Unavailable")

                st.warning(
                    "The rating model could not be trained because "
                    "there are not enough usable historical ratings."
                )

            st.caption(
                "Predictions are estimates based on historical data, "
                "not guaranteed delivery times or actual customer feedback."
            )


st.divider()

st.caption(
    "Statistical Inference Lab | Results depend on the loaded dataset. "
    "Machine-learning predictions are estimates based on historical data."
)
