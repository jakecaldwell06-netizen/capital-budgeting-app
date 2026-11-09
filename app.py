# =============================================================================
# Capital Budgeting Decision Lab
# FINAN 623 - Building Apps with AI, Streamlit, and GitHub
#
# The finance question this app answers:
#   "Should we invest in this project, and if we can only pick one,
#    which project should we pick?"
#
# What the app does:
#   1. Takes yearly cash flows for one or two projects (Year 0 = cost)
#   2. Calculates NPV, IRR, MIRR, Profitability Index, Payback,
#      Discounted Payback, and Equivalent Annual Annuity (EAA)
#   3. Gives an accept / reject decision (and picks a winner for
#      mutually exclusive projects)
#   4. Draws the NPV profile and marks the IRR and crossover rate
#   5. Runs a two-way sensitivity table (discount rate x cash flow change)
#   6. Shows every formula with the numbers plugged in, plus the matching
#      Excel formulas so the results can be checked in a spreadsheet
#
# How to run it on your own computer:
#   pip install -r requirements.txt
#   streamlit run app.py
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1: Imports
#   math      - built into Python (no install needed)
#   altair    - charts (comes with Streamlit, listed in requirements anyway)
#   numpy     - math on lists of numbers; used to solve for IRR
#   pandas    - tables (DataFrames)
#   streamlit - turns this script into a web app
# -----------------------------------------------------------------------------
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# SECTION 2: Example projects (presets)
# Each preset fills in the inputs so a user can see the app work right away.
#   - The Kansas farm example is hypothetical (made-up but realistic numbers)
#   - The textbook example has well-known answers, so it doubles as a check
#   - The strip mine example is an edge case with TWO IRRs
# -----------------------------------------------------------------------------
PRESETS = {
    "Kansas farm: grain bin vs. used combine": {
        "description": (
            "A hypothetical Kansas wheat farm can afford only one big purchase "
            "this year. A grain bin with a dryer lasts 10 years; a used combine "
            "lasts 5. The last year of each includes the resale (salvage) value."
        ),
        "name_a": "Grain bin & dryer",
        "name_b": "Used combine",
        "cf_a": [-240000] + [42000] * 9 + [72000],
        "cf_b": [-150000] + [40000] * 4 + [75000],
        "rate": 8.0,
        "compare": True,
        "exclusive": True,
    },
    "Textbook check: Projects S and L": {
        "description": (
            "A classic textbook pair at a 10% cost of capital, with answers you can "
            "check in Excel: NPV(S) = $78.82, NPV(L) = $100.40, IRR(S) = 14.49%, "
            "IRR(L) = 13.55%, crossover rate = 11.97%. NPV and IRR disagree here."
        ),
        "name_a": "Project S",
        "name_b": "Project L",
        "cf_a": [-1000, 500, 400, 300, 100],
        "cf_b": [-1000, 100, 300, 400, 675],
        "rate": 10.0,
        "compare": True,
        "exclusive": True,
    },
    "Edge case: strip mine (two IRRs)": {
        "description": (
            "A mine costs $1.6M, earns $10M in Year 1, then needs $10M of "
            "clean-up in Year 2 (in $ thousands). Because the cash flows change "
            "sign twice, there are two IRRs (25% and 400%). MIRR and NPV still work."
        ),
        "name_a": "Strip mine",
        "name_b": "Project B",
        "cf_a": [-1600, 10000, -10000],
        "cf_b": [-1600, 1000, 1000],
        "rate": 10.0,
        "compare": False,
        "exclusive": False,
    },
    "Blank template (enter your own)": {
        "description": "Enter your own cash flows. Year 0 is usually the cost (a negative number).",
        "name_a": "Project A",
        "name_b": "Project B",
        "cf_a": [-10000, 3000, 3000, 3000, 3000, 3000],
        "cf_b": [-10000, 0, 0, 0, 0, 0],
        "rate": 10.0,
        "compare": False,
        "exclusive": False,
    },
}


# -----------------------------------------------------------------------------
# SECTION 3: Finance functions
# Each function does ONE calculation. Rates are decimals (10% = 0.10).
# Cash flows are a list where position 0 is Year 0, position 1 is Year 1, ...
# -----------------------------------------------------------------------------
def npv(rate, cash_flows):
    """Net Present Value: sum of CF_t / (1 + r)^t for t = 0..n."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


def sign_changes(cash_flows):
    """Count how many times the cash flows switch between + and - (zeros skipped)."""
    signs = [1 if cf > 0 else -1 for cf in cash_flows if cf != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def all_irrs(cash_flows):
    """
    Find EVERY Internal Rate of Return (rates where NPV = 0).

    Trick: substitute x = 1 / (1 + r). Then NPV becomes a polynomial:
        CF_0 + CF_1*x + CF_2*x^2 + ... + CF_n*x^n = 0
    numpy.roots solves the polynomial. Each real, positive root x gives
    an IRR of r = 1/x - 1. This finds multiple IRRs when they exist,
    which a single "guess"-based IRR (like Excel's) can miss.
    """
    if sign_changes(cash_flows) == 0:
        return []  # no sign change = no IRR exists
    coefficients = list(reversed(cash_flows))  # numpy wants highest power first
    roots = np.roots(coefficients)
    irrs = []
    for root in roots:
        if abs(root.imag) < 1e-9 and root.real > 1e-12:
            rate = 1 / root.real - 1
            if rate > -0.9999:
                irrs.append(rate)
    # Remove near-duplicates and sort from smallest to largest
    unique = []
    for rate in sorted(irrs):
        if not unique or abs(rate - unique[-1]) > 1e-7:
            unique.append(rate)
    return unique


def mirr(cash_flows, finance_rate, reinvest_rate):
    """
    Modified IRR (same method as Excel's =MIRR):
      - Discount negative cash flows to Year 0 at the finance rate
      - Compound positive cash flows to Year n at the reinvestment rate
      - MIRR = (FV of inflows / PV of outflows)^(1/n) - 1
    """
    n = len(cash_flows) - 1
    pv_outflows = sum(cf / (1 + finance_rate) ** t for t, cf in enumerate(cash_flows) if cf < 0)
    fv_inflows = sum(cf * (1 + reinvest_rate) ** (n - t) for t, cf in enumerate(cash_flows) if cf > 0)
    if n < 1 or pv_outflows == 0 or fv_inflows == 0:
        return None
    return (fv_inflows / -pv_outflows) ** (1 / n) - 1


def profitability_index(rate, cash_flows):
    """PI = PV of future cash flows (Years 1..n) / initial cost (Year 0)."""
    if cash_flows[0] >= 0:
        return None
    pv_future = npv(rate, cash_flows) - cash_flows[0]
    return pv_future / -cash_flows[0]


def payback_period(cash_flows, rate=None):
    """
    Years until the cumulative cash flow turns positive.
    If a rate is given, cash flows are discounted first (Discounted Payback).
    Fractional years assume cash arrives evenly during the year.
    Uses the LAST time the running total turns positive, so a project that
    pays back and then goes negative again (e.g., a big clean-up cost) is
    not counted as paid back too early.
    Returns None if there is no up-front cost, math.inf if never paid back.
    """
    if cash_flows[0] >= 0:
        return None
    values = [cf if rate is None else cf / (1 + rate) ** t for t, cf in enumerate(cash_flows)]
    cumulative = list(np.cumsum(values))
    if cumulative[-1] < 0:
        return math.inf
    # Last year that still ends with a negative running total
    last_negative = max(t for t, total in enumerate(cumulative) if total < 0)
    t = last_negative + 1
    return last_negative + (-cumulative[last_negative] / values[t])


def equivalent_annual_annuity(rate, cash_flows):
    """
    EAA: turns NPV into an equal yearly amount over the project's life.
    Used to compare projects with DIFFERENT lives.
      EAA = NPV x r / (1 - (1 + r)^-n)
    """
    n = len(cash_flows) - 1
    value = npv(rate, cash_flows)
    if rate == 0:
        return value / n
    return value * rate / (1 - (1 + rate) ** -n)


def crossover_rates(cf_a, cf_b):
    """Rates where both projects have the same NPV (= IRRs of A minus B)."""
    length = max(len(cf_a), len(cf_b))
    a = list(cf_a) + [0] * (length - len(cf_a))
    b = list(cf_b) + [0] * (length - len(cf_b))
    difference = [x - y for x, y in zip(a, b)]
    return [r for r in all_irrs(difference) if r > 0]


def analyze(cash_flows, rate, reinvest_rate):
    """Run every calculation for one project and return a dictionary of results."""
    irrs = all_irrs(cash_flows)
    return {
        "npv": npv(rate, cash_flows),
        "irrs": irrs,
        "mirr": mirr(cash_flows, rate, reinvest_rate),
        "pi": profitability_index(rate, cash_flows),
        "payback": payback_period(cash_flows),
        "disc_payback": payback_period(cash_flows, rate),
        "eaa": equivalent_annual_annuity(rate, cash_flows),
        "life": len(cash_flows) - 1,
        "sign_changes": sign_changes(cash_flows),
    }


# -----------------------------------------------------------------------------
# SECTION 4: Input checking (error handling)
# Returns (cash_flows, errors, warnings). Errors stop the analysis;
# warnings are shown but the math still runs.
# -----------------------------------------------------------------------------
def clean_cash_flows(column, name):
    """Check one project's column from the input table."""
    errors, warnings = [], []
    values = []
    for year, value in enumerate(column):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            errors.append(f"{name}: Year {year} is blank. Enter 0 if there is no cash flow that year.")
        else:
            values.append(float(value))
    if errors:
        return None, errors, warnings
    if all(v == 0 for v in values):
        errors.append(f"{name}: every cash flow is 0, so there is nothing to analyze.")
        return None, errors, warnings
    if values[0] >= 0:
        warnings.append(
            f"{name}: Year 0 is not negative. Most projects start with a cost "
            "(negative number). Payback and Profitability Index need one, so they show N/A."
        )
    if all(v >= 0 for v in values):
        warnings.append(f"{name}: there are no negative cash flows, so NPV is positive at any rate and there is no IRR.")
    if sign_changes(values) > 1:
        warnings.append(
            f"{name}: cash flows change sign {sign_changes(values)} times, so there can be "
            "more than one IRR. Use NPV or MIRR to decide."
        )
    return values, errors, warnings


# -----------------------------------------------------------------------------
# SECTION 5: Formatting helpers (how numbers look on screen)
# -----------------------------------------------------------------------------
def money(value):
    """$12,345.67 or -$12,345.67"""
    sign = "-" if value < -0.005 else ""
    return f"{sign}${abs(value):,.2f}"


def pct(value):
    return "N/A" if value is None else f"{value * 100:,.2f}%"


def irr_text(irrs):
    if not irrs:
        return "N/A"
    return " and ".join(pct(r) for r in irrs)


def years_text(value):
    if value is None:
        return "N/A"
    if value == math.inf:
        return "Never"
    return f"{value:,.2f} yrs"


# -----------------------------------------------------------------------------
# SECTION 6: Charts
# -----------------------------------------------------------------------------
def npv_profile_chart(projects, rate, max_rate):
    """Line chart of NPV at many discount rates, one line per project."""
    rates = np.linspace(0, max_rate, 201)
    rows = []
    for name, cfs, _ in projects:
        for r in rates:
            rows.append({"Discount rate": r, "NPV": npv(r, cfs), "Project": name})
    data = pd.DataFrame(rows)

    lines = alt.Chart(data).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Discount rate:Q", axis=alt.Axis(format="%"), title="Discount rate"),
        y=alt.Y("NPV:Q", axis=alt.Axis(format="$,.0f"), title="NPV"),
        color=alt.Color("Project:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(range=["#512888", "#E07A1F"])),  # K-State purple, orange
        tooltip=[
            "Project:N",
            alt.Tooltip("Discount rate:Q", format=".2%"),
            alt.Tooltip("NPV:Q", format="$,.2f"),
        ],
    )
    zero_line = alt.Chart(pd.DataFrame({"NPV": [0.0]})).mark_rule(color="gray").encode(y="NPV:Q")
    rate_line = alt.Chart(pd.DataFrame({"Discount rate": [rate]})).mark_rule(
        strokeDash=[6, 4], color="gray").encode(x="Discount rate:Q")

    # Dots at each IRR (where a line crosses zero) and at the crossover rate
    points = []
    for name, _, result in projects:
        for r in result["irrs"]:
            if 0 <= r <= max_rate:
                points.append({"Discount rate": r, "NPV": 0.0, "Label": f"IRR {pct(r)}", "Project": name})
    if len(projects) == 2:
        for r in crossover_rates(projects[0][1], projects[1][1]):
            if r <= max_rate:
                points.append({"Discount rate": r, "NPV": npv(r, projects[0][1]),
                               "Label": f"Crossover {pct(r)}", "Project": "Crossover"})
    layers = [zero_line, rate_line, lines]
    if points:
        point_data = pd.DataFrame(points)
        dots = alt.Chart(point_data).mark_point(size=80, filled=True, color="black").encode(
            x="Discount rate:Q", y="NPV:Q",
            tooltip=["Label:N", alt.Tooltip("NPV:Q", format="$,.2f")],
        )
        labels = dots.mark_text(dy=-12, fontSize=11, color="black").encode(text="Label:N")
        layers += [dots, labels]
    return alt.layer(*layers).properties(height=380)


def sensitivity_table(cash_flows, rate):
    """
    Two-way data table (like Excel's Data Table): NPV for different
    discount rates (rows) and changes to Years 1..n cash flows (columns).
    """
    rate_steps = [rate + d for d in (-0.04, -0.02, 0.0, 0.02, 0.04) if rate + d > -0.99]
    cf_changes = [-0.20, -0.10, 0.0, 0.10, 0.20]
    rows = []
    for r in rate_steps:
        for change in cf_changes:
            adjusted = [cash_flows[0]] + [cf * (1 + change) for cf in cash_flows[1:]]
            rows.append({
                "Discount rate": pct(r),
                "Cash flow change": f"{change:+.0%}",
                "NPV": npv(r, adjusted),
                "order_r": r,
                "order_c": change,
            })
    return pd.DataFrame(rows)


def sensitivity_chart(table):
    rate_order = list(table.sort_values("order_r")["Discount rate"].unique())
    change_order = list(table.sort_values("order_c")["Cash flow change"].unique())
    base = alt.Chart(table).encode(
        x=alt.X("Cash flow change:N", sort=change_order, title="Change to Year 1+ cash flows",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Discount rate:N", sort=rate_order),
    )
    limit = max(abs(table["NPV"].min()), abs(table["NPV"].max()), 1)
    heat = base.mark_rect().encode(
        color=alt.Color("NPV:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-limit, limit]), legend=None),
        tooltip=["Discount rate:N", "Cash flow change:N", alt.Tooltip("NPV:Q", format="$,.2f")],
    )
    text = base.mark_text(fontSize=12).encode(text=alt.Text("NPV:Q", format="$,.0f"))
    return (heat + text).properties(height=260)


# -----------------------------------------------------------------------------
# SECTION 7: The app's screen (everything the user sees)
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Capital Budgeting Decision Lab", page_icon="📈", layout="wide")
    ss = st.session_state

    st.title("📈 Capital Budgeting Decision Lab")
    st.write(
        "Should you take on a project, and if you can only pick one, which one? "
        "Enter yearly cash flows and get **NPV, IRR, MIRR, Profitability Index, Payback,** "
        "and a clear decision."
    )

    # ---------- 7a. Sidebar: settings ----------
    with st.sidebar:
        st.header("Settings")
        preset_name = st.selectbox("Start from an example", list(PRESETS.keys()))
        preset = PRESETS[preset_name]
        st.caption(preset["description"])

        rate_pct = st.number_input(
            "Discount rate / cost of capital (%)",
            min_value=-99.0, max_value=500.0, value=float(preset["rate"]), step=0.25,
            key=f"rate_{preset_name}",
            help="The return you could earn elsewhere at similar risk (often the WACC).",
        )
        same_reinvest = st.checkbox("MIRR reinvestment rate = discount rate", value=True, key=f"same_{preset_name}")
        if same_reinvest:
            reinvest_pct = rate_pct
        else:
            reinvest_pct = st.number_input("MIRR reinvestment rate (%)", min_value=-99.0, max_value=500.0,
                                           value=float(preset["rate"]), step=0.25, key=f"re_{preset_name}")
        compare = st.checkbox("Compare a second project (B)", value=preset["compare"], key=f"cmp_{preset_name}")
        exclusive = False
        if compare:
            exclusive = st.radio(
                "How are the projects related?",
                ["Mutually exclusive (pick one)", "Independent (could do both)"],
                index=0 if preset["exclusive"] else 1, key=f"ex_{preset_name}",
            ).startswith("Mutually")

    rate = rate_pct / 100
    reinvest_rate = reinvest_pct / 100

    # ---------- 7b. Cash flow inputs ----------
    st.subheader("1. Enter cash flows")
    st.caption("Year 0 is today (usually the cost, entered as a negative). "
               "Include salvage/resale value in the final year. Use 0 for years with no cash flow.")

    col_a, col_b, col_years = st.columns([2, 2, 1])
    name_a = col_a.text_input("Project A name", preset["name_a"], key=f"na_{preset_name}") or "Project A"
    name_b = (col_b.text_input("Project B name", preset["name_b"], key=f"nb_{preset_name}") or "Project B") if compare else None
    if name_b == name_a:
        name_b = name_a + " (B)"

    # Build the starting table from the preset the first time it is chosen
    preset_len = max(len(preset["cf_a"]), len(preset["cf_b"]))
    if ss.get("loaded_preset") != preset_name:
        ss.loaded_preset = preset_name
        ss.base_table = pd.DataFrame({
            "Year": list(range(preset_len)),
            "A": preset["cf_a"] + [0.0] * (preset_len - len(preset["cf_a"])),
            "B": preset["cf_b"] + [0.0] * (preset_len - len(preset["cf_b"])),
        }).astype({"A": float, "B": float})
        ss.latest_table = ss.base_table.copy()

    years = col_years.number_input("Years (after Year 0)", min_value=1, max_value=50,
                                   value=preset_len - 1, step=1, key=f"yrs_{preset_name}")

    # If the number of years changed, grow or shrink the table (keeping edits)
    if years + 1 != len(ss.base_table):
        old = ss.latest_table
        new = pd.DataFrame({"Year": list(range(years + 1)), "A": 0.0, "B": 0.0})
        keep = min(len(old), years + 1)
        new.loc[: keep - 1, ["A", "B"]] = old.loc[: keep - 1, ["A", "B"]].values
        ss.base_table = new

    column_config = {
        "Year": st.column_config.NumberColumn("Year", disabled=True, format="%d"),
        "A": st.column_config.NumberColumn(f"{name_a} ($)", format="dollar", step=0.01),
        "B": st.column_config.NumberColumn(f"{name_b or 'Project B'} ($)", format="dollar", step=0.01),
    }
    shown_columns = ["Year", "A", "B"] if compare else ["Year", "A"]
    edited = st.data_editor(
        ss.base_table,
        column_config=column_config,
        column_order=shown_columns,
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        height=min(36 * (len(ss.base_table) + 1) + 3, 760),  # show every year without scrolling
        key=f"editor_{preset_name}_{len(ss.base_table)}",
    )
    ss.latest_table = edited

    # ---------- 7c. Check inputs ----------
    cf_a, errors, warnings = clean_cash_flows(list(edited["A"]), name_a)
    cf_b = None
    if compare:
        cf_b, errors_b, warnings_b = clean_cash_flows(list(edited["B"]), name_b)
        errors += errors_b
        warnings += warnings_b
    for message in errors:
        st.error(message)
    if errors:
        st.stop()
    for message in warnings:
        st.warning(message)
    if rate > 1:
        st.warning("A discount rate above 100% is unusual. Double-check the rate.")

    # Projects don't need the same life: trim trailing zero years for each
    def trim(cfs):
        end = len(cfs)
        while end > 2 and cfs[end - 1] == 0:
            end -= 1
        return cfs[:end]

    cf_a = trim(cf_a)
    projects = [(name_a, cf_a, analyze(cf_a, rate, reinvest_rate))]
    if compare:
        cf_b = trim(cf_b)
        projects.append((name_b, cf_b, analyze(cf_b, rate, reinvest_rate)))

    # ---------- 7d. Decision ----------
    st.subheader("2. Decision")
    show_decision(projects, rate, exclusive)

    # ---------- 7e. Results tabs ----------
    st.subheader("3. Analysis")
    tab_metrics, tab_profile, tab_detail, tab_sens, tab_formulas = st.tabs(
        ["📊 Metrics", "📉 NPV profile", "🧾 Year-by-year", "🎛️ Sensitivity", "🧮 Formulas & Excel check"]
    )

    with tab_metrics:
        show_metrics(projects, rate, compare)

    with tab_profile:
        finite_irrs = [r for _, _, res in projects for r in res["irrs"]]
        auto_max = max([0.20, rate * 2] + [r * 1.25 for r in finite_irrs])
        auto_max = min(auto_max, 5.0)
        max_rate = st.slider("Chart up to discount rate (%)", 5, 500, int(math.ceil(auto_max * 100)),
                             key=f"maxr_{preset_name}") / 100
        st.altair_chart(npv_profile_chart(projects, rate, max_rate), width="stretch")
        st.caption("Dashed line = your discount rate. Where a line crosses $0 is that project's IRR. "
                   "If the lines cross each other, that rate is the crossover rate.")
        if compare:
            cross = crossover_rates(cf_a, cf_b)
            if cross:
                st.info(
                    f"**Crossover rate: {irr_text(cross)}.** Below it, the project with more of its cash "
                    "later (or larger scale) has the higher NPV; above it, the other project wins. "
                    "This is why NPV and IRR can disagree."
                )
            else:
                st.info("The NPV lines do not cross at a positive rate, so NPV and IRR rank the projects the same way.")

    with tab_detail:
        for name, cfs, _ in projects:
            st.markdown(f"**{name}**")
            st.dataframe(year_by_year(cfs, rate), hide_index=True, width="stretch")

    with tab_sens:
        st.write("How NPV changes if the discount rate or the yearly cash flows are different from your estimate. "
                 "Green = positive NPV (accept), red = negative (reject).")
        pick = st.radio("Project", [p[0] for p in projects], horizontal=True, key=f"sens_{preset_name}")
        cfs = next(c for n, c, _ in projects if n == pick)
        table = sensitivity_table(cfs, rate)
        st.altair_chart(sensitivity_chart(table), width="stretch")
        base = npv(rate, cfs)
        worst = table["NPV"].min()
        st.caption(f"Base NPV {money(base)}. Worst case in this table (highest rate, -20% cash flows): {money(worst)}.")

    with tab_formulas:
        show_formulas(projects, rate, reinvest_rate)

    # ---------- 7f. Download ----------
    st.divider()
    summary = pd.DataFrame([summary_row(n, c, r, rate) for n, c, r in projects])
    st.download_button("⬇️ Download results (CSV)", summary.to_csv(index=False).encode("utf-8"),
                       file_name="capital_budgeting_results.csv", mime="text/csv")
    st.caption("Educational tool. Cash flow estimates drive everything; results are only as good as the inputs.")


# -----------------------------------------------------------------------------
# SECTION 8: Pieces of the screen (called from main)
# -----------------------------------------------------------------------------
def show_decision(projects, rate, exclusive):
    """Plain-English accept / reject decision."""
    if len(projects) == 1 or not exclusive:
        for name, _, res in projects:
            if res["npv"] > 0.005:
                st.success(f"✅ **Accept {name}.** NPV is {money(res['npv'])} at {pct(rate)}: "
                           "it earns more than the cost of capital and adds value.")
            elif res["npv"] < -0.005:
                st.error(f"❌ **Reject {name}.** NPV is {money(res['npv'])} at {pct(rate)}: "
                         "it earns less than the cost of capital.")
            else:
                st.info(f"➖ **{name} breaks even** (NPV ≈ $0). It earns exactly the cost of capital.")
        return

    (name_a, _, a), (name_b, _, b) = projects
    if a["npv"] <= 0 and b["npv"] <= 0:
        st.error(f"❌ **Reject both.** Neither project has a positive NPV at {pct(rate)}.")
        return

    npv_winner = name_a if a["npv"] >= b["npv"] else name_b
    if a["life"] == b["life"]:
        st.success(f"✅ **Choose {npv_winner}.** It has the higher NPV "
                   f"({money(max(a['npv'], b['npv']))} vs. {money(min(a['npv'], b['npv']))}).")
    else:
        eaa_winner = name_a if a["eaa"] >= b["eaa"] else name_b
        st.success(
            f"✅ **Choose {eaa_winner}** (highest Equivalent Annual Annuity). The projects last "
            f"{a['life']} vs. {b['life']} years, so EAA compares them on a per-year basis: "
            f"{name_a} {money(a['eaa'])}/yr vs. {name_b} {money(b['eaa'])}/yr."
        )
        if eaa_winner != npv_winner:
            st.warning(
                f"Heads up: {npv_winner} has the bigger total NPV, but it takes longer to earn it. "
                f"If the shorter project could be repeated when it ends, {eaa_winner} creates more "
                f"value per year. If it can't be repeated, go with the higher NPV ({npv_winner})."
            )

    # Does IRR agree with NPV? (only meaningful when the lives match)
    if a["life"] == b["life"] and len(a["irrs"]) == 1 and len(b["irrs"]) == 1:
        irr_winner = name_a if a["irrs"][0] >= b["irrs"][0] else name_b
        if irr_winner != npv_winner:
            st.info(f"IRR prefers {irr_winner}, but NPV prefers {npv_winner}. When they conflict, "
                    "trust NPV: it measures dollars of value added (see the crossover rate on the NPV profile tab).")


def show_metrics(projects, rate, compare):
    """Metric cards and a comparison table."""
    columns = st.columns(len(projects))
    for column, (name, cfs, res) in zip(columns, projects):
        with column:
            st.markdown(f"#### {name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("NPV", money(res["npv"]))
            c2.metric("IRR", irr_text(res["irrs"]))
            c3.metric("MIRR", pct(res["mirr"]))
            c4, c5, c6 = st.columns(3)
            c4.metric("Profitability Index", "N/A" if res["pi"] is None else f"{res['pi']:.3f}")
            c5.metric("Payback", years_text(res["payback"]))
            c6.metric("Disc. payback", years_text(res["disc_payback"]))
            st.caption(f"Life: {res['life']} years · EAA: {money(res['eaa'])}/yr")
            if len(res["irrs"]) > 1:
                st.warning(f"Multiple IRRs ({irr_text(res['irrs'])}). IRR can't be used to decide here.")

    st.markdown("##### Decision rules")
    st.markdown(
        f"""
| Measure | Accept if… | Why |
|---|---|---|
| NPV | NPV > $0 | Adds dollars of value at the {pct(rate)} cost of capital |
| IRR | IRR > {pct(rate)} | Project's own return beats the cost of capital |
| MIRR | MIRR > {pct(rate)} | Like IRR, but assumes realistic reinvestment and has only one answer |
| Profitability Index | PI > 1.0 | Each $1 invested returns more than $1 in present value |
| Payback / Disc. payback | Shorter is better | Measures how fast money comes back (ignores later cash flows) |
"""
    )


def year_by_year(cash_flows, rate):
    rows = []
    cumulative, cumulative_pv = 0.0, 0.0
    for t, cf in enumerate(cash_flows):
        factor = 1 / (1 + rate) ** t
        cumulative += cf
        cumulative_pv += cf * factor
        rows.append({
            "Year": t,
            "Cash flow": money(cf),
            "Discount factor 1/(1+r)^t": f"{factor:.4f}",
            "Present value": money(cf * factor),
            "Cumulative cash flow": money(cumulative),
            "Cumulative PV": money(cumulative_pv),
        })
    return pd.DataFrame(rows)


def show_formulas(projects, rate, reinvest_rate):
    st.markdown("**The formulas**")
    st.latex(r"NPV = \sum_{t=0}^{n} \frac{CF_t}{(1+r)^t}")
    st.latex(r"IRR:\ \sum_{t=0}^{n} \frac{CF_t}{(1+IRR)^t} = 0")
    st.latex(r"MIRR = \left(\frac{\sum CF^{+}_t (1+r_{re})^{n-t}}{\left|\sum CF^{-}_t / (1+r)^t\right|}\right)^{1/n} - 1")
    st.latex(r"PI = \frac{PV(CF_1 \ldots CF_n)}{|CF_0|} \qquad EAA = NPV \times \frac{r}{1-(1+r)^{-n}}")

    for name, cfs, res in projects:
        st.markdown(f"---\n**{name}: NPV with your numbers**")
        terms = " + ".join(f"\\frac{{{cf:,.0f}}}{{(1{rate:+.4f})^{{{t}}}}}" for t, cf in enumerate(cfs[:6]))
        if len(cfs) > 6:
            terms += r" + \cdots"
        st.latex(f"NPV = {terms} = {res['npv']:,.2f}")

        last_row = len(cfs) + 1
        st.markdown("**Check it in Excel** (put the cash flows in B2:B{0}, Year 0 in B2):".format(last_row))
        st.code(
            f"NPV  =NPV({rate:.4f}, B3:B{last_row}) + B2        → {money(res['npv'])}\n"
            f"IRR  =IRR(B2:B{last_row})                          → {irr_text(res['irrs'])}\n"
            f"MIRR =MIRR(B2:B{last_row}, {rate:.4f}, {reinvest_rate:.4f})        → {pct(res['mirr'])}",
            language=None,
        )
    st.info("Excel trap: =NPV() treats its first value as Year 1. That's why Year 0 (B2) is added "
            "outside the function instead of inside the range.")


def summary_row(name, cash_flows, res, rate):
    return {
        "Project": name,
        "Discount rate": rate,
        "Cash flows": "; ".join(f"{cf:g}" for cf in cash_flows),
        "NPV": round(res["npv"], 2),
        "IRR": irr_text(res["irrs"]),
        "MIRR": pct(res["mirr"]),
        "Profitability Index": None if res["pi"] is None else round(res["pi"], 4),
        "Payback (yrs)": years_text(res["payback"]),
        "Discounted payback (yrs)": years_text(res["disc_payback"]),
        "EAA": round(res["eaa"], 2),
    }


# -----------------------------------------------------------------------------
# SECTION 9: Start the app
# Streamlit runs this file as "__main__". The check also lets the finance
# functions above be imported into a test script without drawing the page.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
