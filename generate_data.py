"""
Generate a synthetic but realistic FMCG sales dataset.

The raw sales file is intentionally "dirty" (duplicates, missing values,
inconsistent text, returns) so that the analysis script has real cleaning work to do.

Usage:
    python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).parent / "data"

BRANDS = {
    "Laundry": ["Ariel", "Tide", "Persil"],
    "Hair Care": ["Pantene", "Head & Shoulders", "Elseve"],
    "Oral Care": ["Oral-B", "Blend-a-med", "Colgate"],
    "Shaving": ["Gillette", "Venus", "BIC"],
    "Baby Care": ["Pampers", "Huggies", "Merries"],
}
BASE_PRICE = {"Laundry": 650, "Hair Care": 420, "Oral Care": 260, "Shaving": 890, "Baby Care": 1250}
CITIES = {
    "Moscow": "Central", "Tula": "Central", "Yaroslavl": "Central",
    "Saint Petersburg": "North-West", "Kaliningrad": "North-West",
    "Kazan": "Volga", "Samara": "Volga", "Nizhny Novgorod": "Volga",
    "Yekaterinburg": "Ural", "Chelyabinsk": "Ural",
    "Novosibirsk": "Siberia", "Krasnoyarsk": "Siberia",
}
FORMATS = {"Hypermarket": 3.0, "Supermarket": 1.6, "Convenience": 0.7}
PROMO_SENSITIVITY = {"Laundry": 2.1, "Hair Care": 1.7, "Oral Care": 1.5, "Shaving": 1.3, "Baby Care": 1.2}
BRAND_STRENGTH = [1.4, 1.0, 0.7]  # leader, challenger, follower within each category


def make_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    sku = 1000
    for category, brands in BRANDS.items():
        for brand in brands:
            for size in ["S", "M", "L"]:
                factor = {"S": 0.7, "M": 1.0, "L": 1.5}[size]
                price = BASE_PRICE[category] * factor * rng.uniform(0.85, 1.2)
                rows.append({
                    "brand_strength": BRAND_STRENGTH[brands.index(brand)],
                    "sku": f"SKU{sku}",
                    "product_name": f"{brand} {category} {size}",
                    "brand": brand,
                    "category": category,
                    "list_price": round(price, 0),
                })
                sku += 1
    return pd.DataFrame(rows)


def make_stores(rng: np.random.Generator, n_stores: int = 40) -> pd.DataFrame:
    cities = rng.choice(list(CITIES), size=n_stores)
    formats = rng.choice(list(FORMATS), size=n_stores, p=[0.25, 0.45, 0.30])
    return pd.DataFrame({
        "store_id": [f"S{i:03d}" for i in range(1, n_stores + 1)],
        "city": cities,
        "region": [CITIES[c] for c in cities],
        "format": formats,
        "open_date": pd.to_datetime("2018-01-01")
        + pd.to_timedelta(rng.integers(0, 2000, n_stores), unit="D"),
    })


def make_sales(rng, products, stores) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    n = 60_000
    date = rng.choice(dates, n)
    store = stores.sample(n, replace=True, random_state=SEED).reset_index(drop=True)
    prod = products.sample(n, replace=True, random_state=SEED + 1).reset_index(drop=True)

    month = pd.DatetimeIndex(date).month
    season = 1 + 0.15 * np.sin((month - 3) / 12 * 2 * np.pi)          # mild seasonality
    december = np.where(month == 12, 1.35, 1.0)                        # New Year peak
    fmt = store["format"].map(FORMATS).to_numpy()
    promo = rng.random(n) < 0.18
    sensitivity = prod["category"].map(PROMO_SENSITIVITY).to_numpy()
    promo_uplift = np.where(promo, sensitivity * rng.uniform(0.9, 1.1, n), 1.0)
    discount = np.where(promo, rng.choice([0.1, 0.15, 0.2, 0.25], n), 0.0)

    lam = 4 * season * december * fmt * promo_uplift * prod["brand_strength"].to_numpy()
    units = rng.poisson(lam) + 1
    price = prod["list_price"].to_numpy() * (1 - discount)

    sales = pd.DataFrame({
        "transaction_id": np.arange(1, n + 1),
        "date": pd.DatetimeIndex(date).strftime("%Y-%m-%d"),
        "store_id": store["store_id"],
        "sku": prod["sku"],
        "units": units,
        "unit_price": price.round(2),
        "promo": np.where(promo, "Y", "N"),
    })

    # --- inject typical data quality problems -----------------------------
    idx = rng.choice(n, 400, replace=False)
    sales.loc[idx[:150], "units"] = -sales.loc[idx[:150], "units"]      # returns
    sales.loc[idx[150:300], "unit_price"] = np.nan                      # missing prices
    sales.loc[idx[300:], "promo"] = rng.choice(["yes", "no", " Y", None], 100)  # messy flags
    dup = sales.sample(300, random_state=SEED)                          # duplicated rows
    sales = pd.concat([sales, dup]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return sales


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    products = make_products(rng)
    stores = make_stores(rng)
    sales = make_sales(rng, products, stores)

    # messy city names in the store reference table
    stores.loc[stores.sample(6, random_state=1).index, "city"] = (
        stores["city"].str.upper() + " "
    )

    products = products.drop(columns="brand_strength")
    products.to_csv(DATA_DIR / "products.csv", index=False)
    stores.to_csv(DATA_DIR / "stores.csv", index=False)
    sales.to_csv(DATA_DIR / "sales_raw.csv", index=False)
    print(f"products: {len(products)}, stores: {len(stores)}, sales rows: {len(sales)}")


if __name__ == "__main__":
    main()
