# TAB 7: MACHINE LEARNING PREDICTION
with tabs[6]:
    st.subheader("Food Delivery Time and Customer Rating Prediction")
    st.write(
        "Random Forest regression models predict delivery time "
        "and customer ratings using historical order data."
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

    delivery_data = delivery_data.dropna(
        subset=[
            "delivery_time",
            "food_preparation_time",
            "cost_of_the_order",
            "cuisine_type",
            "day_of_the_week"
        ]
    )

    delivery_data = delivery_data[
        delivery_data["delivery_time"] > 0
    ]

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
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]),
                categorical_features
            )
        ])

    delivery_model = None
    rating_model = None

    if len(delivery_data) < 10:
        st.warning(
            "At least 10 valid orders are required "
            "to train the delivery-time model."
        )

    else:
        # DELIVERY TIME MODEL

        X_delivery = delivery_data[delivery_features]
        y_delivery = delivery_data["delivery_time"]

        delivery_preprocessor = make_preprocessor(
            [
                "food_preparation_time",
                "cost_of_the_order"
            ],
            [
                "cuisine_type",
                "day_of_the_week"
            ]
        )

        delivery_model = Pipeline([
            ("preprocessing", delivery_preprocessor),
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
            y_test,
            delivery_predictions
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

        rating_data = rating_data.dropna(
            subset=rating_features + ["rating_numeric"]
        )

        rating_data = rating_data[
            rating_data["rating_numeric"].between(1, 5)
            & (rating_data["delivery_time"] > 0)
        ]

        if len(rating_data) < 10:
            st.warning(
                f"Only {len(rating_data)} usable customer ratings "
                "are available. At least 10 rated orders are required."
            )

        else:
            X_rating = rating_data[rating_features]
            y_rating = rating_data["rating_numeric"]

            rating_preprocessor = make_preprocessor(
                [
                    "food_preparation_time",
                    "delivery_time",
                    "cost_of_the_order"
                ],
                [
                    "cuisine_type",
                    "day_of_the_week"
                ]
            )

            rating_model = Pipeline([
                ("preprocessing", rating_preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        min_samples_leaf=2
                    )
                )
            ])

            Xr_train, Xr_test, yr_train, yr_test = train_test_split(
                X_rating,
                y_rating,
                test_size=0.2,
                random_state=42
            )

            rating_model.fit(Xr_train, yr_train)

            rating_predictions = rating_model.predict(Xr_test)

            rating_predictions = np.clip(
                rating_predictions, 1, 5
            )

            rating_mae = mean_absolute_error(
                yr_test,
                rating_predictions
            )

            rating_rmse = np.sqrt(
                mean_squared_error(yr_test, rating_predictions)
            )

            rating_r2 = (
                r2_score(yr_test, rating_predictions)
                if len(yr_test) >= 2 else np.nan
            )

            st.write(
                f"Orders used for rating training and evaluation: "
                f"{len(rating_data):,}"
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
                "Lower MAE and RMSE indicate smaller prediction errors. "
                "R² can be negative when the model performs poorly."
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

        # PREDICT FOR A NEW ORDER

        st.divider()
        st.markdown("### Predict for a New Order")

        cuisines = sorted(
            delivery_data["cuisine_type"].astype(str).unique().tolist()
        )

        days = sorted(
            delivery_data["day_of_the_week"].astype(str).unique().tolist()
        )

        with st.form("prediction_form"):
            cuisine = st.selectbox(
                "Cuisine Type",
                cuisines
            )

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

            day = st.selectbox(
                "Day of the Week",
                days
            )

            submitted = st.form_submit_button(
                "Predict Delivery Time and Customer Rating"
            )

        if submitted and delivery_model is not None:

            new_delivery_order = pd.DataFrame([{
                "food_preparation_time": prep_time,
                "cost_of_the_order": cost,
                "cuisine_type": cuisine,
                "day_of_the_week": day
            }])

            predicted_time = float(
                delivery_model.predict(new_delivery_order)[0]
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
                    rating_model.predict(new_rating_order)[0]
                )

                predicted_rating = float(
                    np.clip(predicted_rating, 1, 5)
                )

                b.metric(
                    "Predicted Customer Rating",
                    f"{predicted_rating:.2f} / 5 ⭐"
                )

                st.success(
                    f"Estimated delivery time: {predicted_time:.1f} minutes\n\n"
                    f"Predicted customer rating: {predicted_rating:.2f}/5"
                )

            else:
                b.metric(
                    "Predicted Customer Rating",
                    "Unavailable"
                )

                st.warning(
                    "There are not enough usable historical ratings "
                    "to train the customer rating model."
                )

            st.caption(
                "These are estimates based on historical data, "
                "not guaranteed delivery times or actual customer feedback."
            )


st.divider()

st.caption(
    "Statistical Inference Lab | Statistical results are computed "
    "from the loaded dataset. Predictions depend on the quality "
    "and representativeness of historical data."
)
