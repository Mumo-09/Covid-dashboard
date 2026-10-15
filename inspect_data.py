import os

import pandas as pd

FILE = "WHO-COVID-19-global-daily-data.csv"

# Look in the current folder, then in a /data subfolder
path = next((p for p in [FILE, os.path.join("data", FILE)] if os.path.exists(p)), None)
if path is None:
    raise SystemExit(f"Could not find {FILE}. Put it next to this script or in a /data folder.")

print(f"Reading: {path}\n")
df = pd.read_csv(path)

print("=== SHAPE ===")
print(f"{df.shape[0]:,} rows x {df.shape[1]} columns\n")

print("=== COLUMNS & TYPES ===")
print(df.dtypes.to_string(), "\n")

print("=== FIRST 5 ROWS ===")
print(df.head().to_string(), "\n")

print("=== MISSING VALUES (columns with any) ===")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if missing.empty:
    print("None")
else:
    print(pd.DataFrame({"missing": missing, "percent": (missing / len(df) * 100).round(1)}).to_string())
print()

print("=== NUMERIC SUMMARY ===")
print(df.describe().T.to_string(), "\n")

# WHO file uses Date_reported, Country, WHO_region, New_cases, Cumulative_cases, ...
if "Date_reported" in df.columns:
    dates = pd.to_datetime(df["Date_reported"], errors="coerce")
    print("=== DATE RANGE ===")
    print(f"{dates.min().date()} -> {dates.max().date()}  (unparseable dates: {dates.isna().sum()})\n")

if "Country" in df.columns:
    print("=== COUNTRIES ===")
    print(f"{df['Country'].nunique()} unique countries/areas")
    print("Sample:", ", ".join(map(str, df["Country"].dropna().unique()[:8])), "\n")

if "WHO_region" in df.columns:
    print("=== WHO REGIONS ===")
    print(df["WHO_region"].value_counts().to_string(), "\n")

if {"Country", "Cumulative_cases", "Cumulative_deaths"} <= set(df.columns):
    print("=== TOP 10 COUNTRIES BY CUMULATIVE CASES ===")
    latest = df.groupby("Country")[["Cumulative_cases", "Cumulative_deaths"]].max()
    print(latest.sort_values("Cumulative_cases", ascending=False).head(10).to_string(), "\n")

print(f"Duplicate rows: {df.duplicated().sum():,}")