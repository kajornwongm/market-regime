"""
RS Rotation Signal — rs_signal.py
Run every end-of-month to compute RS signal and update data/rs_latest.json

Usage:
    python rs_signal.py

Output:
    data/rs_latest.json   ← loaded by index.html
    data/rs_YYYYMMDD.json ← snapshot archive

Strategy:
    L1: 14 country ETFs + TLT + IEF + GLD vs VT → hold top 2
    L2: 11 sector ETFs vs SPY → drill down when SPY = L1 signal
    Binary filter: if signal clarity score < 5.5 → cash 100%
    Cost model: Webull 0.107% buy / ~0.109% sell
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Output paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUT_LATEST   = DATA_DIR / "rs_latest.json"
OUT_SNAPSHOT = DATA_DIR / f"rs_{datetime.utcnow().strftime('%Y%m%d')}.json"
PREV_PATH    = DATA_DIR / "rs_previous.json"

# ══════════════════════════════════════════════════════
# CONFIG — edit here if universe/params change
# ══════════════════════════════════════════════════════
L1_TICKERS = {
    "SPY":  "🇺🇸 US",
    "EWJ":  "🇯🇵 Japan",
    "EWY":  "🇰🇷 Korea",
    "INDA": "🇮🇳 India",
    "EWG":  "🇩🇪 Germany",
    "EWZ":  "🇧🇷 Brazil",
    "EWT":  "🇹🇼 Taiwan",
    "VNM":  "🇻🇳 Vietnam",
    "MCHI": "🇨🇳 China",
    "EWA":  "🇦🇺 Australia",
    "EWU":  "🇬🇧 UK",
    "EWQ":  "🇫🇷 France",
    "KSA":  "🇸🇦 Saudi Arabia",
    "EWW":  "🇲🇽 Mexico",
    "TLT":  "🟡 US 20yr Bond",
    "IEF":  "🟡 US 7-10yr Bond",
    "GLD":  "🥇 Gold",
}

L2_TICKERS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLY":  "Cons. Discretionary",
    "XLP":  "Cons. Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLC":  "Communication",
    "TLT":  "🟡 US 20yr Bond",
    "GLD":  "🥇 Gold",
}

ASSET_CLASS = {
    "TLT": "bond", "IEF": "bond",
    "GLD": "gold", "IAU": "gold",
}

L1_BENCHMARK  = "VT"
L2_BENCHMARK  = "SPY"
L1_TOP_N      = 2
L2_TOP_N      = 3
SCORE_THRESH  = 5.5   # signal clarity threshold → cash if below

# RS computation params
LOOKBACK_MONTHS = 11   # months of momentum (skip 1)
SKIP_MONTHS     = 1
W_PRICE         = 0.65
W_VOLUME        = 0.35
VOL_SHORT_WEEKS = 4
VOL_LONG_WEEKS  = 13

# Webull cost
WEBULL_BUY  = 0.00107
WEBULL_SELL = 0.00107
SEC_RATE    = 0.0000206
FINRA_EXTRA = 0.0003   # approx FINRA TAF + CAT


# ══════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════
def download_data(start_date="2014-01-01"):
    all_tickers = (
        list(L1_TICKERS.keys())
        + list(L2_TICKERS.keys())
        + [L1_BENCHMARK, L2_BENCHMARK]
    )
    all_tickers = list(dict.fromkeys(all_tickers))  # dedupe

    print(f"Downloading {len(all_tickers)} tickers from {start_date}...")
    raw = yf.download(
        all_tickers, start=start_date,
        end=datetime.today().strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False
    )
    price  = raw["Close"].ffill().dropna(how="all")
    volume = raw["Volume"].ffill().dropna(how="all")

    # Apply daily cap ±25% to remove anomalies
    DAILY_CAP = 0.25
    price_clean = price.copy()
    capped = []
    for col in price.columns:
        dr = price[col].pct_change()
        if (dr.abs() > DAILY_CAP).any():
            dr_c = dr.clip(-DAILY_CAP, DAILY_CAP)
            dr_c.iloc[0] = 0
            fp = price_clean[col].first_valid_index()
            price_clean[col] = price_clean.loc[fp, col] * (1 + dr_c).cumprod()
            capped.append(col)
    if capped:
        print(f"  Capped anomalies in: {capped}")

    print(f"  Data: {price_clean.index[0].date()} → {price_clean.index[-1].date()}")
    return price_clean, volume


# ══════════════════════════════════════════════════════
# RS SCORE COMPUTATION
# ══════════════════════════════════════════════════════
def calc_price_rs(price, tickers, benchmark, lookback=11, skip=1):
    """Price momentum ratio: (1+ETF_ret) / (1+BM_ret) over lookback months."""
    monthly = price.resample("ME").last()
    tickers = [t for t in tickers if t in monthly.columns]
    records = []
    for i in range(lookback + skip, len(monthly)):
        date   = monthly.index[i]
        t_skip = i - skip
        t_past = t_skip - lookback
        if t_past < 0:
            continue
        p_skip = monthly.iloc[t_skip]
        p_past = monthly.iloc[t_past]
        if benchmark not in monthly.columns:
            continue
        bm_ret = (p_skip[benchmark] / p_past[benchmark]) - 1
        row = {}
        for t in tickers:
            if t not in monthly.columns:
                continue
            etf_ret = (p_skip[t] / p_past[t]) - 1
            row[t] = (1 + etf_ret) / (1 + bm_ret) if bm_ret != 0 else (1 + etf_ret)
        records.append({"date": date, **row})
    return pd.DataFrame(records).set_index("date")


def calc_volume_trend(volume, tickers, short_w=4, long_w=13):
    """Short/long vol ratio: >1 = volume expanding."""
    tickers = [t for t in tickers if t in volume.columns]
    s = volume[tickers].rolling(short_w * 5).mean()
    l = volume[tickers].rolling(long_w * 5).mean()
    ratio = (s / l).fillna(1.0)
    return ratio.resample("ME").last()


def combine_rs(price_rs, vol_trend, tickers, w_price=0.65, w_vol=0.35):
    """Weighted composite: price RS + volume trend."""
    common = price_rs.index.intersection(vol_trend.index)
    tickers = [t for t in tickers if t in price_rs.columns and t in vol_trend.columns]
    scores = {}
    for date in common:
        row = {}
        for t in tickers:
            p = float(price_rs.loc[date, t]) if t in price_rs.columns else 1.0
            v = float(vol_trend.loc[date, t]) if t in vol_trend.columns else 1.0
            row[t] = w_price * p + w_vol * v
        scores[date] = row
    return pd.DataFrame(scores).T


# ══════════════════════════════════════════════════════
# SIGNAL CLARITY SCORE
# ══════════════════════════════════════════════════════
def compute_clarity_score(rs_score, vol_trend, holdings, prev_holdings_list,
                           w_margin=0.40, w_consistency=0.35, w_volume=0.25):
    """
    3-component signal clarity score (0–10):
    1. RS margin above benchmark (40%)
    2. Signal consistency streak (35%)
    3. Volume confirmation (25%)
    """
    if not holdings:
        return 0.0, 0.0, 0.0, 0.0

    latest_date = rs_score.index[-1]

    # Component 1: RS margin
    rs_vals = [rs_score.loc[latest_date, t]
               for t in holdings if t in rs_score.columns]
    avg_rs  = float(np.mean(rs_vals)) if rs_vals else 1.0
    margin  = avg_rs - 1.0
    c_margin = float(np.clip(margin / 0.20 * 5 + 5, 0, 10))

    # Component 2: Consistency streak
    streak = 0
    held_set = set(holdings)
    for prev in reversed(prev_holdings_list[-6:]):
        if set(prev) == held_set:
            streak += 1
        else:
            break
    c_consistency = min(3 + streak * 2.33, 10.0)

    # Component 3: Volume confirmation
    if latest_date in vol_trend.index:
        vol_vals = [float(vol_trend.loc[latest_date, t])
                    for t in holdings if t in vol_trend.columns]
        vol_confirm = float(np.mean([1 if v >= 1.0 else 0 for v in vol_vals])) if vol_vals else 0.5
    else:
        vol_confirm = 0.5
    c_volume = vol_confirm * 10

    composite = w_margin * c_margin + w_consistency * c_consistency + w_volume * c_volume
    return round(composite, 2), round(c_margin, 2), round(c_consistency, 2), round(c_volume, 2)


# ══════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ══════════════════════════════════════════════════════
def get_current_signal(l1_rs, l2_rs, prev_holdings_list, vol_trend):
    """
    Compute this month's action from L1 + L2 RS scores.
    Returns full signal dict for dashboard.
    """
    latest = l1_rs.index[-1]
    l1_tickers = [t for t in L1_TICKERS if t in l1_rs.columns]
    l2_tickers = [t for t in L2_TICKERS if t in l2_rs.columns]

    # L1 ranking
    l1_scores = l1_rs.loc[latest, l1_tickers].dropna().sort_values(ascending=False)
    l1_outperform = l1_scores[l1_scores > 1.0]

    # Check clarity score using previous month signal (avoid look-ahead)
    prev_hold = prev_holdings_list[-1] if prev_holdings_list else []
    score, c_margin, c_cons, c_vol = compute_clarity_score(
        l1_rs, vol_trend, prev_hold, prev_holdings_list[:-1]
    )

    # Apply binary filter (threshold 5.5 — hidden from dashboard)
    if score < SCORE_THRESH and len(prev_holdings_list) > 0:
        return {
            "mode": "CASH",
            "reason": f"Signal clarity {score} < threshold {SCORE_THRESH}",
            "l1_top": [],
            "final_holdings": [],
            "sells": [],
            "buys": [],
            "l2_active": False,
            "clarity_score": score,
            "clarity_components": {"margin": c_margin, "consistency": c_cons, "volume": c_vol},
        }

    if len(l1_outperform) == 0:
        return {
            "mode": "CASH",
            "reason": "All L1 assets underperform benchmark",
            "l1_top": [],
            "final_holdings": [],
            "sells": [],
            "buys": [],
            "l2_active": False,
            "clarity_score": score,
            "clarity_components": {"margin": c_margin, "consistency": c_cons, "volume": c_vol},
        }

    l1_top = list(l1_outperform.head(L1_TOP_N).index)

    # L2 drill-down when SPY in L1 top
    final_holdings = []
    l2_active = False
    l2_sectors = []
    slot_weight = 1.0 / L1_TOP_N

    for ticker in l1_top:
        if ticker == L2_BENCHMARK and latest in l2_rs.index:
            # Drill into sectors
            l2_scores = l2_rs.loc[latest, l2_tickers].dropna().sort_values(ascending=False)
            l2_out = l2_scores[l2_scores > 1.0]
            top_secs = list(l2_out.head(L2_TOP_N).index) if len(l2_out) > 0 else [ticker]
            sec_w = slot_weight / len(top_secs)
            for s in top_secs:
                final_holdings.append({"ticker": s, "weight": round(sec_w, 4), "layer": "L2"})
            l2_active = True
            l2_sectors = top_secs
        else:
            final_holdings.append({"ticker": ticker, "weight": round(slot_weight, 4), "layer": "L1"})

    # Compute sells / buys vs previous holdings
    prev_set = set(prev_hold)
    new_set  = set(h["ticker"] for h in final_holdings)
    sells = list(prev_set - new_set)
    buys  = list(new_set - prev_set)
    holds = list(prev_set & new_set)

    # Full L1 ranking for RS data tab
    l1_full = [
        {
            "ticker": t,
            "name": L1_TICKERS.get(t, t),
            "rs": round(float(l1_rs.loc[latest, t]), 4),
            "asset_class": ASSET_CLASS.get(t, "equity"),
            "outperforms": bool(l1_rs.loc[latest, t] > 1.0),
        }
        for t in l1_tickers if t in l1_rs.columns
    ]
    l1_full.sort(key=lambda x: x["rs"], reverse=True)

    # Full L2 ranking
    l2_full = []
    if latest in l2_rs.index:
        l2_full = [
            {
                "ticker": t,
                "name": L2_TICKERS.get(t, t),
                "rs": round(float(l2_rs.loc[latest, t]), 4),
                "outperforms": bool(l2_rs.loc[latest, t] > 1.0),
            }
            for t in l2_tickers if t in l2_rs.columns
        ]
        l2_full.sort(key=lambda x: x["rs"], reverse=True)

    # Heatmap: last 6 months
    def make_heatmap(rs_df, tickers_map):
        n_months = min(6, len(rs_df))
        recent   = rs_df.iloc[-n_months:]
        months   = [d.strftime("%b %y") for d in recent.index]
        rows = []
        for t in tickers_map:
            if t not in rs_df.columns:
                continue
            vals = [round(float(v), 2) for v in recent[t].fillna(1.0).values]
            rows.append({"ticker": t, "name": tickers_map[t], "values": vals})
        return {"months": months, "rows": rows}

    l1_hm = make_heatmap(l1_rs, L1_TICKERS)
    l2_hm = make_heatmap(l2_rs, L2_TICKERS) if not l2_rs.empty else {"months": [], "rows": []}

    return {
        "mode": "ACTIVE",
        "l1_top": l1_top,
        "final_holdings": final_holdings,
        "sells": sells,
        "buys":  buys,
        "holds": holds,
        "l2_active": l2_active,
        "l2_sectors": l2_sectors,
        "l1_ranking": l1_full,
        "l2_ranking": l2_full,
        "l1_heatmap": l1_hm,
        "l2_heatmap": l2_hm,
        "clarity_score": score,
        "clarity_components": {
            "margin":      c_margin,
            "consistency": c_cons,
            "volume":      c_vol,
            "avg_rs":      round(avg_rs, 4),
            "streak":      int(streak),
            "threshold":   SCORE_THRESH,
        },
    }


# ══════════════════════════════════════════════════════
# COST ESTIMATE
# ══════════════════════════════════════════════════════
def estimate_cost(sells, buys, portfolio_size, n_slots):
    """Webull round-trip cost estimate."""
    if not sells and not buys:
        return 0.0
    trade_val = portfolio_size / n_slots
    sell_cost = len(sells) * trade_val * (WEBULL_SELL + SEC_RATE + FINRA_EXTRA)
    buy_cost  = len(buys)  * trade_val * WEBULL_BUY
    return round(sell_cost + buy_cost, 2)


# ══════════════════════════════════════════════════════
# DRAWDOWN CONTEXT
# ══════════════════════════════════════════════════════
def compute_dd_context(price, ticker="VT"):
    """Compute current drawdown + historical context."""
    if ticker not in price.columns:
        return {}
    s = price[ticker].dropna()
    peak   = s.cummax()
    dd_ser = (s - peak) / peak * 100
    current_dd = round(float(dd_ser.iloc[-1]), 2)
    max_dd     = round(float(dd_ser.min()), 2)

    # Count historical DDs of similar size
    dd_monthly = s.resample("ME").last().pct_change().dropna()
    threshold  = current_dd * 0.7 if current_dd < 0 else -3.0
    similar    = int((dd_monthly < threshold / 100).sum())

    # Duration: how many months since last peak
    peak_monthly = s.resample("ME").last().cummax()
    s_monthly    = s.resample("ME").last()
    below_peak   = (s_monthly < peak_monthly).values
    duration = 0
    for v in reversed(below_peak):
        if v:
            duration += 1
        else:
            break

    return {
        "current":    current_dd,
        "max":        max_dd,
        "duration_m": duration,
        "similar_10y": similar,
        "severity_pct": round(current_dd / max_dd * 100, 1) if max_dd < 0 else 0,
    }


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    print(f"RS Signal — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # Load previous signal for delta comparison
    prev = {}
    if PREV_PATH.exists():
        try:
            prev = json.loads(PREV_PATH.read_text())
            print(f"  Previous signal: {prev.get('date','—')} | "
                  f"holdings: {[h['ticker'] for h in prev.get('final_holdings',[])]}")
        except Exception:
            pass
    prev_holdings_list = prev.get("holdings_history", [])

    # 1. Download
    price, volume = download_data()

    # 2. Compute RS scores
    l1_tickers = [t for t in L1_TICKERS if t in price.columns]
    l2_tickers = [t for t in L2_TICKERS if t in price.columns]

    print("Computing L1 RS (country+TLT+GLD vs VT)...")
    l1_price_rs  = calc_price_rs(price, l1_tickers, L1_BENCHMARK, LOOKBACK_MONTHS, SKIP_MONTHS)
    l1_vol_trend = calc_volume_trend(volume, l1_tickers, VOL_SHORT_WEEKS, VOL_LONG_WEEKS)
    l1_rs        = combine_rs(l1_price_rs, l1_vol_trend, l1_tickers, W_PRICE, W_VOLUME)

    print("Computing L2 RS (sectors vs SPY)...")
    l2_price_rs  = calc_price_rs(price, l2_tickers, L2_BENCHMARK, LOOKBACK_MONTHS, SKIP_MONTHS)
    l2_vol_trend = calc_volume_trend(volume, l2_tickers, VOL_SHORT_WEEKS, VOL_LONG_WEEKS)
    l2_rs        = combine_rs(l2_price_rs, l2_vol_trend, l2_tickers, W_PRICE, W_VOLUME)

    vol_trend_combined = pd.concat([l1_vol_trend, l2_vol_trend], axis=1)
    vol_trend_combined = vol_trend_combined.loc[:, ~vol_trend_combined.columns.duplicated()]

    # 3. Generate signal
    print("Generating signal...")
    signal = get_current_signal(l1_rs, l2_rs, prev_holdings_list, vol_trend_combined)

    # 4. Cost estimate (assumes $30k Satellite — adjust as needed)
    PORTFOLIO_SIZE = 30000
    est_cost = estimate_cost(
        signal.get("sells", []),
        signal.get("buys",  []),
        PORTFOLIO_SIZE, L1_TOP_N
    )

    # 5. DD context
    dd_ctx = compute_dd_context(price, L1_BENCHMARK)

    # 6. Assemble output
    current_holdings = [h["ticker"] for h in signal.get("final_holdings", [])]
    holdings_history = (prev_holdings_list + [current_holdings])[-13:]  # keep 13m

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "date":         datetime.today().strftime("%Y-%m-%d"),
        "signal":       signal,
        "est_cost_usd": est_cost,
        "dd_context":   dd_ctx,
        "portfolio_size": PORTFOLIO_SIZE,
        "holdings_history": holdings_history,
        "prev_holdings":  prev.get("final_holdings", []),
        "config": {
            "l1_top_n": L1_TOP_N,
            "l2_top_n": L2_TOP_N,
            "benchmark_l1": L1_BENCHMARK,
            "benchmark_l2": L2_BENCHMARK,
            "score_threshold": SCORE_THRESH,
            "lookback_months": LOOKBACK_MONTHS,
        },
    }

    # 7. Save
    OUT_LATEST.write_text(json.dumps(output, indent=2))
    OUT_SNAPSHOT.write_text(json.dumps(output, indent=2))
    PREV_PATH.write_text(json.dumps({
        "date": output["date"],
        "final_holdings": signal.get("final_holdings", []),
        "holdings_history": holdings_history,
    }))

    # 8. Print summary
    print(f"\n{'='*55}")
    print(f"  Mode     : {signal['mode']}")
    if signal["mode"] == "ACTIVE":
        print(f"  L1 top   : {signal['l1_top']}")
        print(f"  Holdings : {[h['ticker']+' '+str(round(h['weight']*100,1))+'%' for h in signal['final_holdings']]}")
        print(f"  L2 active: {signal['l2_active']}" + (f" → {signal['l2_sectors']}" if signal['l2_active'] else ""))
        print(f"  SELL     : {signal['sells'] or '—'}")
        print(f"  BUY      : {signal['buys'] or '—'}")
    print(f"  Clarity  : {signal['clarity_score']} ({'PASS' if signal['clarity_score'] >= SCORE_THRESH else 'FAIL — cash'})")
    print(f"  Est cost : ${est_cost}")
    print(f"  DD now   : {dd_ctx.get('current','—')}%  (max {dd_ctx.get('max','—')}%,  {dd_ctx.get('duration_m','—')}m)")
    print(f"{'='*55}")
    print(f"  Saved → {OUT_LATEST}")


if __name__ == "__main__":
    main()
