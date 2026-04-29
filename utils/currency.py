import logging
import time
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Fallback rates in case live API is unavailable
FALLBACK_RATES = {
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

# Real-time FX rate cache with 1-hour TTL
_fx_cache: Dict[str, tuple] = {}
_FX_CACHE_TTL_SECONDS = 3600


def _fetch_live_rates() -> Dict[str, float]:
    """Fetch live exchange rates from free API. Falls back to static rates on error."""
    try:
        # Use exchangerate-api.com free tier or fallback to local rates
        # For production, consider using Open Exchange Rates or similar with auth
        response = requests.get(
            "https://v6.exchangerate-api.com/v6/latest/USD",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("result") != "success":
            logger.info("Using fallback FX rates (live API unavailable)")
            return FALLBACK_RATES
        
        rates = data.get("conversion_rates", {})
        if not rates:
            logger.info("Using fallback FX rates (empty response)")
            return FALLBACK_RATES
        
        # Always include USD
        rates["USD"] = 1.0
        logger.info(f"Fetched live exchange rates for {len(rates)} currencies")
        return rates
    
    except requests.exceptions.RequestException as e:
        logger.info(f"Live FX service unavailable: {e}. Using fallback rates.")
        return FALLBACK_RATES
    except Exception as e:
        logger.error(f"Unexpected error fetching exchange rates: {e}")
        return FALLBACK_RATES


def _get_cached_rates() -> Dict[str, float]:
    """Get cached FX rates with automatic refresh on TTL expiry."""
    now = time.time()
    
    if "rates" in _fx_cache:
        cached_time, cached_rates = _fx_cache["rates"]
        if now - cached_time < _FX_CACHE_TTL_SECONDS:
            return cached_rates
    
    # Fetch new rates
    rates = _fetch_live_rates()
    _fx_cache["rates"] = (now, rates)
    return rates


def resolve_currency(region_code: Optional[str], currency_code: Optional[str]) -> str:
    normalized_currency = (currency_code or "").strip().upper()
    all_rates = _get_cached_rates()
    if normalized_currency in all_rates:
        return normalized_currency

    normalized_region = (region_code or "").strip().upper()
    if normalized_region in REGION_TO_CURRENCY:
        return REGION_TO_CURRENCY[normalized_region]

    return "USD"


def convert_from_usd(amount_usd: float, currency_code: str) -> float:
    rates = _get_cached_rates()
    rate = rates.get(currency_code.upper(), 1.0)
    return round(amount_usd * rate, 2)


def convert_to_usd(amount: float, currency_code: str) -> float:
    rates = _get_cached_rates()
    rate = rates.get(currency_code.upper(), 1.0)
    if rate <= 0:
        rate = 1.0
    return round(amount / rate, 2)
