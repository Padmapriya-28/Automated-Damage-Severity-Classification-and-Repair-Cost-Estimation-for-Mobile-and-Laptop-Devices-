from typing import Optional

# Static rates relative to USD. Replace with a live FX source in production.
USD_TO_CURRENCY = {
    "USD": 1.0,
    "INR": 83.2,
    "EUR": 0.92,
    "GBP": 0.79,
    "AED": 3.67,
    "JPY": 149.0,
    "AUD": 1.53,
    "CAD": 1.35,
    "SGD": 1.34,
}

REGION_TO_CURRENCY = {
    "US": "USD",
    "IN": "INR",
    "GB": "GBP",
    "DE": "EUR",
    "FR": "EUR",
    "IT": "EUR",
    "ES": "EUR",
    "NL": "EUR",
    "AE": "AED",
    "JP": "JPY",
    "AU": "AUD",
    "CA": "CAD",
    "SG": "SGD",
}


def resolve_currency(region_code: Optional[str], currency_code: Optional[str]) -> str:
    normalized_currency = (currency_code or "").strip().upper()
    if normalized_currency in USD_TO_CURRENCY:
        return normalized_currency

    normalized_region = (region_code or "").strip().upper()
    if normalized_region in REGION_TO_CURRENCY:
        return REGION_TO_CURRENCY[normalized_region]

    return "USD"


def convert_from_usd(amount_usd: float, currency_code: str) -> float:
    rate = USD_TO_CURRENCY.get(currency_code.upper(), 1.0)
    return round(amount_usd * rate, 2)


def convert_to_usd(amount: float, currency_code: str) -> float:
    rate = USD_TO_CURRENCY.get(currency_code.upper(), 1.0)
    if rate <= 0:
        rate = 1.0
    return round(amount / rate, 2)
