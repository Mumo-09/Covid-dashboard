import glob

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="COVID-19 Dashboard", page_icon="🦠", layout="wide")
st.title("🦠 COVID-19 Dashboard")

# ---------- 1. Load data ----------
csv_files = glob.glob("*.csv") + glob.glob("data/*.csv")
uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type="csv")

if uploaded is not None:
    source = uploaded
elif csv_files:
    source = st.sidebar.selectbox("Data file", csv_files)
else:
    st.warning("No CSV found. Put your file next to app.py (or in a /data folder), or upload one in the sidebar.")
    st.stop()


@st.cache_data
def load(src):
    return pd.read_csv(src)


raw = load(source)
cols = list(raw.columns)

# ---------- 2a. Data inspector (same checks as inspect_data.py) ----------
with st.expander("🔍 Inspect the data"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw.shape[0]:,}")
    c2.metric("Columns", raw.shape[1])
    c3.metric("Duplicate rows", f"{raw.duplicated().sum():,}")

    tab_head, tab_types, tab_missing, tab_stats = st.tabs(
        ["First rows", "Columns & types", "Missing values", "Numeric summary"]
    )
    with tab_head:
        st.dataframe(raw.head(20), use_container_width=True)
    with tab_types:
        st.dataframe(
            pd.DataFrame({"column": raw.columns, "type": raw.dtypes.astype(str).values,
                          "unique values": raw.nunique().values}),
            use_container_width=True,
        )
    with tab_missing:
        miss = raw.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if miss.empty:
            st.success("No missing values.")
        else:
            st.dataframe(
                pd.DataFrame({"missing": miss, "percent": (miss / len(raw) * 100).round(1)}),
                use_container_width=True,
            )
    with tab_stats:
        st.dataframe(raw.describe().T, use_container_width=True)


def guess(options):
    """Pick the first column whose name matches one of the options."""
    lower = {c.lower(): c for c in cols}
    for o in options:
        if o.lower() in lower:
            return cols.index(lower[o.lower()])
    return 0


# ---------- 2. Column mapping (auto-guessed, change if wrong) ----------
st.sidebar.header("Columns")
date_col = st.sidebar.selectbox("Date", cols, index=guess(["date", "observationdate", "day"]))
country_col = st.sidebar.selectbox(
    "Country / region", cols,
    index=guess(["location", "country", "country/region", "country_region", "entity"]),
)
cases_col = st.sidebar.selectbox("Cases", cols, index=guess(["total_cases", "confirmed", "cases", "new_cases"]))
deaths_col = st.sidebar.selectbox("Deaths", cols, index=guess(["total_deaths", "deaths", "new_deaths"]))
cumulative = st.sidebar.checkbox(
    "Cases/deaths columns are cumulative totals",
    value=any(k in cases_col.lower() for k in ["total", "confirmed"]) or cases_col.lower() == "cases",
    help="Untick if your columns already hold daily new numbers.",
)

# ---------- 3. Clean ----------
df = raw[[date_col, country_col, cases_col, deaths_col]].copy()
df.columns = ["date", "country", "cases", "deaths"]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "country"])
for c in ["cases", "deaths"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# one row per country per day (handles province-level data)
df = df.groupby(["country", "date"], as_index=False)[["cases", "deaths"]].sum().sort_values(["country", "date"])

if cumulative:
    df["new_cases"] = df.groupby("country")["cases"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby("country")["deaths"].diff().fillna(0).clip(lower=0)
else:
    df["new_cases"] = df["cases"]
    df["new_deaths"] = df["deaths"]

# ---------- 4. Filters ----------
st.sidebar.header("Filters")
all_countries = sorted(df["country"].unique())
default = [c for c in ["United States", "US", "India", "Brazil", "Kenya"] if c in all_countries][:3] or all_countries[:3]
selected = st.sidebar.multiselect("Countries", all_countries, default=default)

min_d, max_d = df["date"].min().date(), df["date"].max().date()
start, end = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)

mask = df["date"].between(pd.to_datetime(start), pd.to_datetime(end))
period = df[mask]
view = period[period["country"].isin(selected)]

if view.empty:
    st.info("Select at least one country to see data.")
    st.stop()

# ---------- 5. KPI cards ----------
total_cases = int(view["new_cases"].sum())
total_deaths = int(view["new_deaths"].sum())
fatality = (total_deaths / total_cases * 100) if total_cases else 0

k1, k2, k3 = st.columns(3)
k1.metric("Cases (selected period)", f"{total_cases:,}")
k2.metric("Deaths (selected period)", f"{total_deaths:,}")
k3.metric("Case fatality rate", f"{fatality:.2f}%")

# ---------- 6. Charts ----------
metric_label = st.radio("Metric", ["Cases", "Deaths"], horizontal=True)
new_col = "new_cases" if metric_label == "Cases" else "new_deaths"

daily = view.copy()
daily["7-day average"] = daily.groupby("country")[new_col].transform(lambda s: s.rolling(7, min_periods=1).mean())

left, right = st.columns(2)

with left:
    fig = px.line(daily, x="date", y="7-day average", color="country",
                  title=f"Daily new {metric_label.lower()} (7-day average)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    cum = view.copy()
    cum["cumulative"] = cum.groupby("country")[new_col].cumsum()
    fig = px.line(cum, x="date", y="cumulative", color="country",
                  title=f"Cumulative {metric_label.lower()} in period")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

top = (period.groupby("country")[new_col].sum().sort_values(ascending=False).head(10).reset_index())
with left:
    fig = px.bar(top, x=new_col, y="country", orientation="h",
                 title=f"Top 10 countries by {metric_label.lower()}")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

world = period.groupby("country")[new_col].sum().reset_index()
with right:
    fig = px.choropleth(world, locations="country", locationmode="country names",
                        color=new_col, color_continuous_scale="Reds",
                        title=f"World map: {metric_label.lower()}")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("View data"):
    st.dataframe(view, use_container_width=True)