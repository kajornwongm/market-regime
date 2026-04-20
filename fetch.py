"""
Market Regime Dashboard — fetch.py v2
======================================
Auto (GitHub Actions ทุกวัน):
  - Price, MA, RS scores จาก yfinance
  - Regime detection + alerts

Manual (วางไฟล์ใน data/flows/ สัปดาห์ละครั้ง):
  - ETFdb CSV แยก asset class: equity_YYYY-MM-DD.csv, bond_YYYY-MM-DD.csv
  - ระบบ detect asset class จากชื่อไฟล์อัตโนมัติ
  - Aggregate flow ต่อ asset class และ region
"""

import json, re, warnings
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
DATA_DIR  = Path("data")
FLOWS_DIR = DATA_DIR / "flows"
DATA_DIR.mkdir(exist_ok=True)
FLOWS_DIR.mkdir(exist_ok=True)

ASSET_CLASS_KEYWORDS = {
    "equity":       "Equity",
    "bond":         "Bond",
    "commodity":    "Commodity",
    "real-estate":  "Real Estate",
    "real_estate":  "Real Estate",
    "currency":     "Currency",
    "alternatives": "Alternatives",
    "multi-asset":  "Multi Asset",
    "multi_asset":  "Multi Asset",
}

REGION_KEYWORDS = {
    "North America":       ["large cap","mid cap","small cap","us equity","s&p","nasdaq","russell","total stock","north america"],
    "Europe":              ["europe","european","eurozone","germany","uk","france","switzerland","nordic","developed europe"],
    "Asia Pacific":        ["asia","japan","china","korea","india","taiwan","southeast","pacific","hong kong","singapore","broad asia"],
    "Latin America":       ["latin","brazil","mexico","latin america"],
    "Emerging Markets":    ["emerging market"],
    "Global / Intl":       ["global","international","world","eafe","developed market","global ex"],
    "Middle East / Africa":["middle east","africa","saudi","israel","mena"],
}

PRICE_TICKERS = [
    "SPY","QQQ","IWM","VTV",
    "TLT","AGG","SHY",
    "GLD","DBC","XLE",
    "EWY","EWJ","INDA","EWZ","VNM","EWT","EWG","EWU",
    "XLK","XLV","XLI","XLU","XAR","XLF","XLB",
    "VNQ","IBIT",
]

def get_prices(lookback_days=200):
    end   = datetime.today()
    start = end - timedelta(days=lookback_days + 10)
    print(f"Downloading {len(PRICE_TICKERS)} tickers...")
    raw = yf.download(PRICE_TICKERS,
                      start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
    return raw["Close"].ffill(), raw["Volume"].ffill()

def clean_dollar(s):
    if pd.isna(s): return 0.0
    s = str(s).replace('"','').replace(',','').replace('$','').strip()
    if s in ('','N/A','-','nan'): return 0.0
    try: return float(s)
    except: return 0.0

def parse_etfdb_csv(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = [c.strip().strip('"') for c in df.columns]
        if "Symbol" not in df.columns:
            return None

        flow_map = {}
        for col in df.columns:
            cl = col.lower()
            if "1 week ff"  in cl: flow_map["flow_1w"]  = col
            if "4 week ff"  in cl: flow_map["flow_4w"]  = col
            if "1 year ff"  in cl: flow_map["flow_1y"]  = col
            if "ytd ff"     in cl: flow_map["flow_ytd"] = col

        for key, col in flow_map.items():
            df[key] = df[col].apply(clean_dollar) / 1e6

        for col in ["flow_1w","flow_4w","flow_1y","flow_ytd"]:
            if col not in df.columns:
                df[col] = 0.0

        cat_col = next((c for c in df.columns if "category" in c.lower()), None)

        result = pd.DataFrame({
            "Symbol":     df["Symbol"].astype(str).str.strip(),
            "name":       df.get("ETF Name", pd.Series("", index=df.index)),
            "asset_raw":  df.get("Asset Class", pd.Series("", index=df.index)).fillna(""),
            "category":   df[cat_col].fillna("") if cat_col else "",
            "flow_1w":    df["flow_1w"],
            "flow_4w":    df["flow_4w"],
            "flow_1y":    df["flow_1y"],
            "flow_ytd":   df["flow_ytd"],
        })
        return result.dropna(subset=["Symbol"])
    except Exception as e:
        print(f"  warn {filepath.name}: {e}")
        return None

def detect_asset_class(stem):
    sl = stem.lower()
    for kw, label in ASSET_CLASS_KEYWORDS.items():
        if kw in sl:
            return label
    return "Unknown"

def assign_region(category):
    cl = str(category).lower()
    for region, kws in REGION_KEYWORDS.items():
        if any(k in cl for k in kws):
            return region
    return "Other"

def read_flows():
    all_dfs, file_dates = [], []

    for f in sorted(FLOWS_DIR.glob("*.csv"), reverse=True):
        df = parse_etfdb_csv(f)
        if df is None or df.empty:
            continue
        df["asset_class"] = detect_asset_class(f.stem)
        all_dfs.append(df)
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.stem)
        if m: file_dates.append(m.group(1))
        print(f"  {f.name} → {df['asset_class'].iloc[0]} ({len(df)} ETFs)")

    if not all_dfs:
        return {}, {}, {}, None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Symbol"], keep="first")

    # Ticker flows
    ticker_flows = {}
    for _, r in combined.iterrows():
        t = r["Symbol"]
        if not t: continue
        ticker_flows[t] = {
            "flow_1w_mn":  round(float(r["flow_1w"]),  1),
            "flow_4w_mn":  round(float(r["flow_4w"]),  1),
            "flow_1y_mn":  round(float(r["flow_1y"]),  1),
            "flow_ytd_mn": round(float(r["flow_ytd"]), 1),
            "direction":   "inflow" if r["flow_4w"] > 0 else "outflow",
            "accelerating":bool(r["flow_1w"] > (r["flow_4w"]/4 + 1)),
        }

    # Asset class aggregate
    asset_flows = {}
    for ac, grp in combined.groupby("asset_class"):
        if ac == "Unknown": continue
        asset_flows[ac] = {
            "flow_1w_mn":  round(float(grp["flow_1w"].sum()),  0),
            "flow_4w_mn":  round(float(grp["flow_4w"].sum()),  0),
            "flow_ytd_mn": round(float(grp["flow_ytd"].sum()), 0),
            "n_etfs":      int(len(grp)),
            "n_inflow":    int((grp["flow_4w"] > 0).sum()),
            "n_outflow":   int((grp["flow_4w"] < 0).sum()),
        }

    # Region aggregate
    region_flows = {}
    eq = combined[combined["asset_raw"].str.lower().str.contains("equity", na=False)].copy()
    if not eq.empty:
        eq["region"] = eq["category"].apply(assign_region)
        for reg, grp in eq.groupby("region"):
            if reg in ("Other",): continue
            region_flows[reg] = {
                "flow_1w_mn":  round(float(grp["flow_1w"].sum()),  0),
                "flow_4w_mn":  round(float(grp["flow_4w"].sum()),  0),
                "flow_ytd_mn": round(float(grp["flow_ytd"].sum()), 0),
                "n_etfs":      int(len(grp)),
            }

    return ticker_flows, asset_flows, region_flows, (max(file_dates) if file_dates else None)

# ── Regime scoring ─────────────────────────────────────────────────────────
def mom(price, t, d=21):
    if t not in price.columns: return 0.0
    s = price[t].dropna()
    if len(s) < d+1: return 0.0
    return float((s.iloc[-1]/s.iloc[-(d+1)]-1)*100)

def ma_sig(price, t, short=50, long=200):
    if t not in price.columns: return "unknown"
    c = price[t].dropna()
    if len(c) < long: return "insufficient"
    ms,ml,p = c.rolling(short).mean().iloc[-1], c.rolling(long).mean().iloc[-1], c.iloc[-1]
    return "bullish" if (p>ml and ms>ml) else ("bearish" if p<ml else "neutral")

def score_growth(p):
    return float(np.clip(50+((mom(p,"QQQ")-mom(p,"SPY"))*0.6+(mom(p,"QQQ")-mom(p,"VTV"))*0.4)*2,0,100))

def score_risk(p):
    return float(np.clip(50+(mom(p,"SPY")-mom(p,"TLT")-max(mom(p,"GLD",10)-3,0))*2,0,100))

def score_inflation(p):
    return float(np.clip(50+(mom(p,"DBC")*0.5+mom(p,"XLE")*0.3-mom(p,"TLT")*0.2)*2,0,100))

def score_liquidity(p):
    return float(np.clip(50+(mom(p,"TLT")*0.5+mom(p,"VNQ")*0.3+mom(p,"SHY")*0.2)*3,0,100))

def score_sectors(p):
    secs = {"XLK":"Technology","XLV":"Healthcare","XLI":"Industrials",
            "XLE":"Energy","XLU":"Utilities","XAR":"Defense","XLF":"Financials","XLB":"Materials"}
    spy_m = mom(p,"SPY")
    ranking = [{"ticker":t,"name":n,"momentum":round(mom(p,t),1),"rs":round(mom(p,t)-spy_m,1)}
               for t,n in secs.items() if t in p.columns]
    ranking.sort(key=lambda x:x["rs"],reverse=True)
    top3 = np.mean([r["rs"] for r in ranking[:3]]) if ranking else 0
    return float(np.clip(50+top3*3,0,100)), ranking

def classify_regime(s):
    c = s["growth"]*0.20+s["risk"]*0.25+s["inflation"]*0.20+s["liquidity"]*0.20+s["sector"]*0.15
    if c>=65:
        label = "Late Cycle / Inflation" if s["inflation"]>65 else ("Risk-On / Growth" if s["growth"]>65 else "Risk-On / Neutral")
    elif c>=45: label="Transition / Neutral"
    else: label="Risk-Off / Tightening" if s["liquidity"]<35 else "Risk-Off / Defensive"
    return round(c,1), label

def get_strategy(r):
    return {"Risk-On / Growth":["Long QQQ","Long XLK","Long EWY/INDA","Avoid TLT"],
            "Risk-On / Neutral":["Long SPY","Moderate QQQ","Watch rotation"],
            "Late Cycle / Inflation":["Long XLE","Long GLD","Long DBC","Reduce QQQ"],
            "Transition / Neutral":["Balanced SPY/TLT","Reduce concentration","Watch alerts"],
            "Risk-Off / Tightening":["Long TLT","Long SHY","Long GLD","Reduce equity"],
            "Risk-Off / Defensive":["Long XLU","Long XLV","Long TLT","Cash buffer"],
            }.get(r, ["Review positions","Monitor signals"])

def compute_countries(price, ticker_flows):
    ctry = {"EWY":"🇰🇷 Korea","EWJ":"🇯🇵 Japan","INDA":"🇮🇳 India",
            "EWZ":"🇧🇷 Brazil","VNM":"🇻🇳 Vietnam","EWT":"🇹🇼 Taiwan",
            "EWG":"🇩🇪 Germany","EWU":"🇬🇧 UK","SPY":"🇺🇸 US"}
    spy_m = mom(price,"SPY")
    result = []
    for t,name in ctry.items():
        fl = ticker_flows.get(t,{})
        result.append({"ticker":t,"name":name,
                        "ret_1m":round(mom(price,t),1),"ret_3m":round(mom(price,t,63),1),
                        "rs_vs_spy":round(mom(price,t)-spy_m,1),
                        "ma_signal":ma_sig(price,t),
                        "flow_4w_mn":fl.get("flow_4w_mn"),
                        "flow_1w_mn":fl.get("flow_1w_mn"),
                        "flow_ytd_mn":fl.get("flow_ytd_mn"),
                        "flow_accel":fl.get("accelerating"),
                        "flow_dir":fl.get("direction")})
    return sorted(result, key=lambda x:x["rs_vs_spy"], reverse=True)

def generate_alerts(scores, sect_ranking, asset_flows, ticker_flows, prev_scores):
    alerts = []
    for name,s in scores.items():
        if s>85: alerts.append({"type":"extreme","level":"warning","msg":f"{name.title()} extreme high ({s:.0f}) — watch reversal"})
        if s<15: alerts.append({"type":"extreme","level":"warning","msg":f"{name.title()} extreme low ({s:.0f}) — watch reversal"})
    if prev_scores:
        for name,s in scores.items():
            d = s-prev_scores.get(name,s)
            if abs(d)>=15:
                alerts.append({"type":"regime_shift","level":"alert","msg":f"Regime shift: {name.title()} {'↑' if d>0 else '↓'} {abs(d):.0f}pt"})
    if scores["risk"]>60 and scores["liquidity"]>60:
        alerts.append({"type":"divergence","level":"info","msg":"Divergence: Risk-on + Easing simultaneously"})
    if sect_ranking:
        top=sect_ranking[0]
        if top["ticker"] not in ["XLK"]:
            alerts.append({"type":"leadership","level":"info","msg":f"Sector leadership: {top['name']} (RS +{top['rs']:.1f}%)"})
    for ac,af in asset_flows.items():
        if abs(af["flow_4w_mn"])>50000:
            alerts.append({"type":"asset_flow","level":"alert","msg":f"{ac} {'+' if af['flow_4w_mn']>0 else ''}{af['flow_4w_mn']/1000:.0f}B (4W)"})
    for t,tf in ticker_flows.items():
        if abs(tf["flow_4w_mn"])>1000 and tf["accelerating"]:
            alerts.append({"type":"ticker_flow","level":"info","msg":f"{t}: {'+' if tf['flow_4w_mn']>0 else ''}{tf['flow_4w_mn']:.0f}M (4W) ↑"})
    return alerts[:12]

def main():
    print(f"Market Regime Fetch v2 — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    prev_path = DATA_DIR/"previous_scores.json"
    prev_scores = json.loads(prev_path.read_text()).get("scores",{}) if prev_path.exists() else {}

    price, vol = get_prices()

    print("\nReading flow CSVs...")
    ticker_flows, asset_flows, region_flows, flow_date = read_flows()
    print(f"  Tickers: {len(ticker_flows)} | Asset classes: {len(asset_flows)} | Regions: {len(region_flows)}")

    sect_score, sect_ranking = score_sectors(price)
    scores = {"growth":score_growth(price),"risk":score_risk(price),
              "inflation":score_inflation(price),"liquidity":score_liquidity(price),"sector":sect_score}
    composite, regime_label = classify_regime(scores)

    output = {
        "generated_at":    datetime.utcnow().isoformat()+"Z",
        "flow_date":        flow_date,
        "flow_updated":     bool(asset_flows or ticker_flows),
        "n_flow_tickers":   len(ticker_flows),
        "n_asset_classes":  len(asset_flows),
        "regime":           {"label":regime_label,"composite":composite,"scores":{k:round(v,1) for k,v in scores.items()}},
        "strategy":         get_strategy(regime_label),
        "alerts":           generate_alerts(scores,sect_ranking,asset_flows,ticker_flows,prev_scores),
        "asset_flows":      asset_flows,
        "region_flows":     region_flows,
        "sectors":          sect_ranking,
        "countries":        compute_countries(price,ticker_flows),
        "ticker_flows":     ticker_flows,
        "ma_signals":       {t:ma_sig(price,t) for t in ["SPY","QQQ","TLT","GLD","EWY","AGG"]},
    }

    (DATA_DIR/"latest.json").write_text(json.dumps(output,indent=2))
    (DATA_DIR/f"snapshot_{datetime.utcnow().strftime('%Y%m%d')}.json").write_text(json.dumps(output,indent=2))
    prev_path.write_text(json.dumps({"scores":scores,"date":datetime.utcnow().isoformat()}))

    print(f"\nRegime: {regime_label} ({composite})")
    if asset_flows:
        print("Asset flows (4W):")
        for ac,af in sorted(asset_flows.items(),key=lambda x:abs(x[1]["flow_4w_mn"]),reverse=True):
            print(f"  {ac:<15} {'+' if af['flow_4w_mn']>0 else ''}{af['flow_4w_mn']/1000:.1f}B")

if __name__ == "__main__":
    main()
