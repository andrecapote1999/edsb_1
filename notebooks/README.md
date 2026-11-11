# Telco Customer Churn: Retention Management

**Enterprise Data Science Bootcamp 2026 (NOVA IMS), Group 2 (`EDSB26_G2`)**

Team: André Capote, João Gonçalves, Paulo Amaro, Tomás Ramos

## Overview

A telecom company has limited retention resources and needs to decide **which existing customers should receive a retention intervention, and in what order**. A churn score on its own is not the goal. This project estimates each customer's churn risk and turns it into a prioritised contact list.

> **Research question:** How can we use customer churn risk estimates to prioritise retention interventions under limited operational capacity?

The work is organised around three problems:

| # | Problem | Where it is answered |
|---|---|---|
| 1 | Which customer characteristics are associated with churn? | Notebook section 1.6, summarised in 6.1 |
| 2 | Which customers are at the highest risk of churn? | Chapters 3 and 4, summarised in 6.2 |
| 3 | Whom should we contact first, given a cap on outreach? | Chapter 5, summarised in 6.3 |

## Data

The five source files come from the course (Telco Customer Churn sample, IBM Cognos Analytics) and are **not included in this repository**.

| File | One row per | Joins on |
|---|---|---|
| `Telco_customer_churn_demographics.xlsx` | customer | `Customer ID` |
| `Telco_customer_churn_location.xlsx` | customer | `Customer ID` |
| `Telco_customer_churn_services.xlsx` | customer | `Customer ID` |
| `Telco_customer_churn_status.xlsx` | customer | `Customer ID` |
| `Telco_customer_churn_population.xlsx` | ZIP code | `Zip Code` |

- 7,043 customers, all in California, in a single Q3 snapshot.
- After the merge the grain stays one row per customer (key uniqueness and row counts are checked in the notebook).
- **Modelling population:** customers with `Customer Status` equal to `Churned` (1,869) or `Stayed` (4,720), which gives 6,589 customers and a churn rate of 28.4%. The 454 `Joined` customers are excluded.
- **Target:** churned (1) vs stayed (0).

## Approach

The notebook follows this sequence:

| Chapter | Content |
|---|---|
| 0. Setup & Data Loading | Libraries, file loading, merge with integrity checks, definition of the analytical population |
| 1. Exploratory Data Analysis | Demographics, location and population, services, status, cross-dataset relationships, and the answer to Problem 1 |
| 2. Feature Selection & Engineering | Exclusion of columns that encode the outcome, only exist after churn, duplicate another column or identify location; candidate engineered variables tested with 5-fold cross-validation on the development set only |
| 3. Baseline Model | Stratified 80/20 split, majority-class baseline, logistic regression, first priority list, data-leakage check |
| 4. Model Comparison | Stratified 60/20/20 train/validation/test split; nine candidate models; winner chosen on validation PR-AUC; test set used once |
| 5. Customer Prioritisation | Top-k contact lists at 5%, 10% and 20%; ranking by expected value (P(churn) × monthly revenue); comparison with the existing SPSS Churn Score |
| 6. Conclusions | Answers to the three problems |

Models compared in chapter 4: baseline, logistic regression, decision tree, random forest, gradient boosting and sigmoid-calibrated gradient boosting, with and without the engineered variables. The main metric is PR-AUC, in line with the prioritisation goal.

## Key results

**Problem 1 (association with churn, exploratory analysis).** Churn is 51.7% on Month-to-Month contracts, 10.9% on One Year and 2.6% on Two Year. It is 77.2% in the first 6 months of tenure and 9.5% after 48 months. These are associations, not causes.

**Problems 2 and 3 (test set, 1,318 customers, 374 churners).** Winning model: calibrated gradient boosting with feature engineering.

| Metric | Value |
|---|---|
| PR-AUC | 0.968 |
| ROC-AUC | 0.986 |
| Precision / Recall (threshold 0.5) | 0.938 / 0.850 |
| F1 / Accuracy | 0.892 / 0.942 |
| Log loss / Brier | 0.140 / 0.044 |

| Customers contacted | Churners caught | Precision | Recall |
|---|---|---|---|
| Top 5% (66) | 66 | 100% | 17.6% |
| Top 10% (132) | 132 | 100% | 35.3% |
| Top 20% (264) | 261 | 98.9% | 69.8% |

A random selection would catch 28.4% of churners at any contact level, so the lift is about 3.5×. Ranking by expected value (churn probability × average monthly revenue) changes the list: only 57 of the 132 customers in the top 10% stay on it, and the monthly revenue of the churners reached rises by 30%.

## Known limitations

- **Data leakage.** The column `Churn Score Range`, derived from the existing SPSS Churn Score, is still among the model features. It dominates the logistic regression coefficients and **inflates every metric above**. The comparison with the SPSS Churn Score is therefore not independent. See section 3.5 of the notebook.
- **Single snapshot.** The data has no longitudinal history, so the evaluation is a customer-level holdout from one quarter.

## How to run

**1. Create an environment and install the dependencies** (Python 3.10 or later; tested with 3.13):

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

**2. Add the data.** Put the five `.xlsx` files in one folder, for example `data/raw/`.

**3. Set the data path.** In the notebook, section *0.2 Paths and File Loading* defines `BASE` and `RAW` per team member. Keep one block, comment out the others, and point `RAW` to the folder with the five files.

**4. Run the notebook.** Open `Project-EDSB26_G2_VersaoAtual.ipynb` in VS Code or JupyterLab, select the `.venv` kernel and use *Run All*. A full run takes about two to three minutes.

## Repository contents

```
.
├── README.md
├── requirements.txt
├── Project-EDSB26_G2_VersaoAtual.ipynb   # full analysis, top to bottom
├── data/raw/                             # place the five .xlsx files here (not included)
└── reports_experiments_churn.csv         # generated by the notebook (experiments table)
```

## Reproducibility

- Random seeds are fixed (`random_state=42` for the splits and models, a fixed seed for the NumPy generator).
- The 60/20/20 split reuses exactly the same test set as the 80/20 split, and the notebook asserts it.
- The test set is used only once, after the winner is chosen on validation.

## Acknowledgements

Dataset provenance: IBM Cognos Analytics, *Telco Customer Churn* sample. Project brief: NOVA IMS, EDSB 2026, *Project D: Telco Customer Churn, Retention Management*.
