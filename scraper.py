import asyncio
import requests
from trafilatura import extract as trafilatura_extract
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from typing import List

# -------------------------
# SCRAPER: static + JS fallback
# -------------------------
def extract_static(url: str, timeout: int = 12):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/123.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "*/*"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        html = resp.text
        text = trafilatura_extract(html)
        if text and len(text.strip()) > 200:
            return text
        # fallback to BS
        soup = BeautifulSoup(html, "html.parser")
        raw = soup.get_text("\n", strip=True)
        if raw and len(raw.strip()) > 200:
            return raw
        return None
    except Exception as e:
        print("extract_static error:", e)
        return None

def extract_js(url: str, timeout: int = 30):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout*1000, wait_until="networkidle")
            html = page.content()
            browser.close()
        text = trafilatura_extract(html)
        if text and len(text.strip()) > 200:
            return text
        soup = BeautifulSoup(html, "html.parser")
        raw = soup.get_text("\n", strip=True)
        if raw and len(raw.strip()) > 200:
            return raw
        return None
    except Exception as e:
        print("extract_js error:", e)
        return None

async def fetch_and_combine(urls: List[str]) -> str:
    parts = []
    # loop = asyncio.get_running_loop() # Unused variable
    for url in urls:
        # Try static quickly (run sync in threadpool)
        text = await asyncio.to_thread(extract_static, url)
        if not text:
            # Try JS
            text = await asyncio.to_thread(extract_js, url)
        if not text:
            parts.append(f"[Could not extract content from: {url}]")
        else:
            # optional: add small header for origin
            parts.append(f"--- SOURCE: {url} ---\n{text}\n")
    combined = "\n\n".join(parts).strip()
    return combined
#saloni code
# import asyncio
# import requests
# from bs4 import BeautifulSoup
# from playwright.sync_api import sync_playwright
# from typing import Dict, List, Optional
# import json

# # ---------------------------------------
# # COMMON HEADERS (important for NSE/BSE)
# # ---------------------------------------
# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
#                   "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
#     "Accept-Language": "en-US,en;q=0.9",
#     "Accept": "*/*",
#     "Connection": "keep-alive"
# }

# # ---------------------------------------
# # HELPER: Safe request
# # ---------------------------------------
# def safe_request(url: str, timeout: int = 15):
#     try:
#         session = requests.Session()
#         session.headers.update(HEADERS)
#         response = session.get(url, timeout=timeout)
#         response.raise_for_status()
#         return response.text
#     except Exception as e:
#         print(f"Request error: {e}")
#         return None


# # ---------------------------------------
# # 1️⃣ Yahoo Finance Scraper
# # ---------------------------------------
# def scrape_yahoo(symbol: str) -> Optional[Dict]:
#     url = f"https://finance.yahoo.com/quote/{symbol}"
#     html = safe_request(url)
#     if not html:
#         return None

#     soup = BeautifulSoup(html, "html.parser")

#     try:
#         price = soup.select_one('fin-streamer[data-field="regularMarketPrice"]').text
#         change = soup.select_one('fin-streamer[data-field="regularMarketChange"]').text
#         percent = soup.select_one('fin-streamer[data-field="regularMarketChangePercent"]').text

#         return {
#             "source": "Yahoo Finance",
#             "symbol": symbol,
#             "price": price,
#             "change": change,
#             "percent_change": percent
#         }
#     except:
#         return None


# # ---------------------------------------
# # 2️⃣ MarketWatch Scraper
# # ---------------------------------------
# def scrape_marketwatch(symbol: str) -> Optional[Dict]:
#     url = f"https://www.marketwatch.com/investing/stock/{symbol}"
#     html = safe_request(url)
#     if not html:
#         return None

#     soup = BeautifulSoup(html, "html.parser")

#     try:
#         price = soup.select_one(".intraday__price .value").text.strip()
#         change = soup.select_one(".change--point--q").text.strip()

#         return {
#             "source": "MarketWatch",
#             "symbol": symbol,
#             "price": price,
#             "change": change
#         }
#     except:
#         return None


# # ---------------------------------------
# # 3️⃣ Moneycontrol Scraper
# # ---------------------------------------
# def scrape_moneycontrol(symbol: str) -> Optional[Dict]:
#     url = f"https://www.moneycontrol.com/india/stockpricequote/{symbol}"
#     html = safe_request(url)
#     if not html:
#         return None

#     soup = BeautifulSoup(html, "html.parser")

#     try:
#         price = soup.select_one(".inprice1").text.strip()
#         change = soup.select_one(".inprice2").text.strip()

#         return {
#             "source": "Moneycontrol",
#             "symbol": symbol,
#             "price": price,
#             "change": change
#         }
#     except:
#         return None


# # ---------------------------------------
# # 4️⃣ Investing.com (Requires JS)
# # ---------------------------------------
# def scrape_investing(symbol: str) -> Optional[Dict]:
#     url = f"https://www.investing.com/equities/{symbol}"
#     try:
#         with sync_playwright() as p:
#             browser = p.chromium.launch(headless=True)
#             page = browser.new_page()
#             page.goto(url, wait_until="networkidle")
#             html = page.content()
#             browser.close()

#         soup = BeautifulSoup(html, "html.parser")
#         price = soup.select_one('[data-test="instrument-price-last"]').text

#         return {
#             "source": "Investing.com",
#             "symbol": symbol,
#             "price": price
#         }
#     except:
#         return None


# # ---------------------------------------
# # 5️⃣ NSE India (JSON API Preferred)
# # ---------------------------------------
# def scrape_nse(symbol: str) -> Optional[Dict]:
#     try:
#         url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
#         session = requests.Session()
#         session.headers.update(HEADERS)
#         session.get("https://www.nseindia.com", headers=HEADERS)
#         response = session.get(url, headers=HEADERS)
#         data = response.json()

#         return {
#             "source": "NSE India",
#             "symbol": symbol,
#             "price": data["priceInfo"]["lastPrice"],
#             "change": data["priceInfo"]["change"],
#             "percent_change": data["priceInfo"]["pChange"]
#         }
#     except:
#         return None


# # ---------------------------------------
# # 6️⃣ BSE India
# # ---------------------------------------
# def scrape_bse(scrip_code: str) -> Optional[Dict]:
#     try:
#         url = f"https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w?scripcode={scrip_code}&flag=1"
#         response = requests.get(url, headers=HEADERS)
#         data = response.json()

#         return {
#             "source": "BSE India",
#             "symbol": scrip_code,
#             "latest_price": data["Data"][-1]["close"]
#         }
#     except:
#         return None


# # ---------------------------------------
# # MASTER FUNCTION
# # ---------------------------------------
# async def fetch_stock_data(symbol: str):
#     tasks = [
#         asyncio.to_thread(scrape_yahoo, symbol),
#         asyncio.to_thread(scrape_marketwatch, symbol),
#         asyncio.to_thread(scrape_moneycontrol, symbol),
#         asyncio.to_thread(scrape_investing, symbol),
#         asyncio.to_thread(scrape_nse, symbol),
#     ]

#     results = await asyncio.gather(*tasks)

#     # Filter None results
#     return [r for r in results if r]


# # ---------------------------------------
# # URL CONTENT EXTRACTOR
# # ---------------------------------------
# def extract_static(url: str, timeout: int = 12):
#     try:
#         resp = requests.get(url, headers=HEADERS, timeout=timeout)
#         html = resp.text
#         soup = BeautifulSoup(html, "html.parser")
#         # Remove script and style elements
#         for tag in soup(["script", "style", "nav", "footer", "header"]):
#             tag.decompose()
#         text = soup.get_text("\n", strip=True)
#         if text and len(text.strip()) > 200:
#             return text
#         return None
#     except Exception as e:
#         print("extract_static error:", e)
#         return None


# def extract_js(url: str, timeout: int = 30):
#     try:
#         with sync_playwright() as p:
#             browser = p.chromium.launch(headless=True)
#             page = browser.new_page()
#             page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
#             html = page.content()
#             browser.close()
#         soup = BeautifulSoup(html, "html.parser")
#         for tag in soup(["script", "style", "nav", "footer", "header"]):
#             tag.decompose()
#         text = soup.get_text("\n", strip=True)
#         if text and len(text.strip()) > 200:
#             return text
#         return None
#     except Exception as e:
#         print("extract_js error:", e)
#         return None


# async def fetch_and_combine(urls: List[str]) -> str:
#     parts = []
#     for url in urls:
#         text = await asyncio.to_thread(extract_static, url)
#         if not text:
#             text = await asyncio.to_thread(extract_js, url)
#         if not text:
#             parts.append(f"[Could not extract content from: {url}]")
#         else:
#             parts.append(f"--- SOURCE: {url} ---\n{text}\n")
#     combined = "\n\n".join(parts).strip()
#     return combined