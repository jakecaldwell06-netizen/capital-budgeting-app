# Capital Budgeting Decision Lab

A Streamlit app that answers a core corporate finance question: **should we invest in this project, and if we can only pick one, which one?**

Built for FINAN 623 (Building Apps with AI, Streamlit, and GitHub).

## What it calculates
- **NPV** (Net Present Value)
- **IRR**, including *every* IRR when cash flows change sign more than once
- **MIRR** (Modified IRR), the same method as Excel's `=MIRR`
- **Profitability Index**, **Payback**, and **Discounted Payback**
- **EAA** (Equivalent Annual Annuity) to compare projects with different lives
- **Crossover rate**, where two projects' NPVs are equal

## Features
- Accept / reject decision in plain English, with a winner for mutually exclusive projects
- NPV profile chart with IRRs and crossover rate marked
- Two-way sensitivity table (discount rate × cash flow change), like an Excel Data Table
- Year-by-year present value breakdown
- Every formula shown with your numbers, plus matching Excel formulas
- Downloadable results (CSV)
- Examples: a Kansas farm (grain bin vs. used combine), a textbook check, and a two-IRR strip mine

## Verified
NPV, IRR, and MIRR match Excel's `=NPV`, `=IRR`, and `=MIRR` for every example.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
