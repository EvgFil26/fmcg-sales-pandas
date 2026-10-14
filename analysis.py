"""
FMCG sales analysis with pandas: clean -> merge -> analyse -> report.

Usage:
    python generate_data.py   # once, creates data/*.csv
    python analysis.py        # writes tables and charts to outputs/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.float_format = "{:,.1f}".format

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


# ---------------------------------------------------------------- 1. load
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(DATA / "sales_raw.csv", parse_dates=["date"])
    products = pd.read_csv(DATA / "products.csv")
    stores = pd.read_csv(DATA / "stores.csv", parse_dates=["open_date"])
    return sales, products, stores


# ---------------------------------------------------------------- 2. clean
def clean(sales: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame):
    report = {"raw_rows": len(sales)}

    sales = sales.drop_duplicates().copy()
    report["duplicates_removed"] = report["raw_rows"] - len(sales)

    # normalise promo flag: 'Y', ' Y', 'yes' -> True; everything else -> False
    sales["promo"] = sales["promo"].fillna("N").str.strip().str.upper().isin(["Y", "YES"])

    # returns are kept in a separate table, sales analysis uses positive units only
    returns = sales[sales["units"] < 0]
    sales = sales[sales["units"] > 0].copy()
    report["returns_rows"] = len(returns)

    # missing price -> median price of the same SKU
    report["missing_prices_filled"] = int(sales["unit_price"].isna().sum())
    sales["unit_price"] = sales["unit_price"].fillna(
        sales.groupby("sku")["unit_price"].transform("median")
    )

    stores["city"] = stores["city"].str.strip().str.title()
    report["clean_rows"] = len(sales)
    return sales, returns, stores, report


# ---------------------------------------------------------------- 3. enrich
def build_dataset(sales, products, stores) -> pd.DataFrame:
    df = (
        sales.merge(products, on="sku", how="left", validate="many_to_one")
             .merge(stores, on="store_id", how="left", validate="many_to_one")
    )
    assert df["category"].notna().all(), "unknown SKU in sales"
    df["revenue"] = df["units"] * df["unit_price"]
    df["month"] = df["date"].dt.to_period("M")
    df["weekday"] = df["date"].dt.day_name()
    return df


# ---------------------------------------------------------------- 4. analyse
def monthly_trend(df):
    m = df.groupby("month").agg(revenue=("revenue", "sum"), units=("units", "sum"))
    m["mom_growth_%"] = (m["revenue"].pct_change() * 100).round(1)
    return m


def category_share(df):
    c = df.groupby("category")["revenue"].sum().sort_values(ascending=False).to_frame()
    c["share_%"] = (c["revenue"] / c["revenue"].sum() * 100).round(1)
    return c


def top_brands(df, n=5):
    return (df.groupby(["category", "brand"])["revenue"].sum()
              .sort_values(ascending=False).head(n).to_frame())


def promo_uplift(df):
    """Average units per transaction with vs. without promo, by category."""
    p = df.pivot_table(index="category", columns="promo", values="units", aggfunc="mean")
    p.columns = ["no_promo", "promo"]
    p["uplift_%"] = ((p["promo"] / p["no_promo"] - 1) * 100).round(1)
    return p.sort_values("uplift_%", ascending=False)


def region_category_pivot(df):
    return df.pivot_table(index="region", columns="category", values="revenue",
                          aggfunc="sum", margins=True, margins_name="Total").round(0)


def abc_stores(df):
    """ABC analysis: A = stores giving the first 80% of revenue, B = next 15%, C = rest."""
    s = df.groupby(["store_id", "city", "format"])["revenue"].sum().sort_values(ascending=False)
    s = s.to_frame()
    s["cum_share"] = s["revenue"].cumsum() / s["revenue"].sum()
    s["abc"] = pd.cut(s["cum_share"], [0, 0.8, 0.95, 1.0], labels=["A", "B", "C"])
    return s


def returns_rate(df, returns, products):
    r = returns.merge(products, on="sku").groupby("category")["units"].sum().abs()
    sold = df.groupby("category")["units"].sum()
    return (r / sold * 100).round(2).rename("returns_%").to_frame()


# ---------------------------------------------------------------- 5. charts
def charts(trend, cat, uplift):
    fig, ax = plt.subplots(figsize=(9, 4))
    (trend["revenue"] / 1e6).plot(ax=ax, marker="o")
    ax.set_title("Monthly revenue, 2025"); ax.set_ylabel("RUB, millions"); ax.set_xlabel("")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "monthly_revenue.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    cat["share_%"].sort_values().plot.barh(ax=ax)
    ax.set_title("Revenue share by category, %"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(OUT / "category_share.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    uplift["uplift_%"].sort_values().plot.barh(ax=ax, color="tab:orange")
    ax.set_title("Promo uplift in units per transaction, %"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(OUT / "promo_uplift.png", dpi=120)
    plt.close("all")


def main():
    OUT.mkdir(exist_ok=True)
    sales, products, stores = load()
    sales, returns, stores, report = clean(sales, products, stores)
    df = build_dataset(sales, products, stores)

    results = {
        "monthly_trend": monthly_trend(df),
        "category_share": category_share(df),
        "top_brands": top_brands(df),
        "promo_uplift": promo_uplift(df),
        "region_category": region_category_pivot(df),
        "abc_stores": abc_stores(df),
        "returns_rate": returns_rate(df, returns, products),
    }
    for name, table in results.items():
        table.to_csv(OUT / f"{name}.csv")
    charts(results["monthly_trend"], results["category_share"], results["promo_uplift"])

    print("Data cleaning report:")
    for k, v in report.items():
        print(f"  {k:<24}{v:>8}")
    print(f"\nTotal revenue: {df['revenue'].sum() / 1e6:,.1f} M RUB")
    abc = results["abc_stores"]["abc"].value_counts().sort_index()
    print(f"ABC stores: {abc.to_dict()}")
    for name in ["category_share", "promo_uplift", "top_brands", "returns_rate"]:
        print(f"\n{name}:\n{results[name]}")


if __name__ == "__main__":
    main()
