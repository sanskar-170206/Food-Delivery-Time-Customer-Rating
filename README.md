# Food Delivery Time & Customer Ratings — Statistical Inference Lab

This project adapts the structure of the reference laptop-price dashboard to the **Statistical Inference Lab** topic:

**Analysis of Food Delivery Time and Customer Ratings**

It uses `food_order.csv` and provides an interactive Streamlit dashboard with descriptive statistics, plots, confidence intervals, t-tests, correlation, and a chi-square test.

## Files

- `app.py` — Streamlit dashboard and statistical analyses
- `food_order.csv` — dataset supplied for the project
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — theme configuration

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Analyses included

- Dataset overview and descriptive statistics
- Delivery-time distribution, rating distribution, boxplot and scatterplot
- 95% confidence interval for mean delivery time
- 95% confidence interval for mean recorded rating
- One-sample t-test against a configurable delivery-time benchmark (default 30 minutes)
- Welch two-sample t-test comparing weekday and weekend delivery times
- Pearson and Spearman correlation between delivery time and recorded customer rating
- Chi-square test of independence between rating category and day of week
- Preview and download of cleaned data

## Data handling and limitations

- The original `rating` column contains `Not given` entries. These are treated as missing, not as zero.
- Rating-based analyses use only rows with numeric ratings.
- The one-sample test's benchmark is an example chosen for the analysis; it is not asserted to be an official service standard.
- Statistical conclusions depend on the dataset, assumptions, and sampling process. Association does not prove causation.
- This is a statistical analysis dashboard, not a machine-learning prediction system.

## Suggested project title

**Analysis of Food Delivery Time and Customer Ratings Using Statistical Inference**
