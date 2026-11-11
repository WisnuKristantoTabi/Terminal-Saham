"""Modul logika analisis (support/resisten, tren, indikator) untuk web app."""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# ====== PENGATURAN ======

SIMPAN_EXCEL = True       # simpan hasil ke Excel
WINDOW = 5                # swing high/low = puncak/lembah dibanding 5 hari kiri & kanan
TOLERANSI = 0.015         # titik yang selisihnya <= 1,5% digabung menjadi satu zona
MIN_SENTUHAN = 2          # level valid jika tersentuh minimal 2 kali
JUMLAH_LEVEL = 3          # tampilkan 3 level terdekat di tiap sisi
STOP_LOSS_PCT = 0.02      # stop loss 2% di bawah support
MA_PENDEK = 20            # moving average jangka pendek
MA_MENENGAH = 50          # moving average jangka menengah
MA_PANJANG = 200          # moving average jangka panjang (butuh data >= 200 hari)
BATAS_RETURN = 0.02       # perubahan harga 20 hari > +2% = naik, < -2% = turun
RSI_PERIODE = 14          # periode RSI
MACD_CEPAT, MACD_LAMBAT, MACD_SINYAL = 12, 26, 9   # parameter MACD standar
BB_PERIODE = 20           # periode Bollinger Band
BB_STD = 2                # lebar band = 2 standar deviasi
VOL_PERIODE = 20          # rata-rata volume 20 hari
VOL_LONJAKAN = 1.5        # volume >= 1,5x rata-rata dianggap besar
# ========================


def ambil_data(ticker, periode):
    df = yf.Ticker(ticker).history(period=periode, interval="1d", auto_adjust=False)
    if df.empty:
        raise ValueError(
            f"Data untuk '{ticker}' tidak ditemukan. Untuk saham BEI gunakan akhiran .JK (contoh: BBCA.JK)."
        )
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.dropna(subset=["high", "low", "close"]).sort_values("date").reset_index(drop=True)


def cari_swing(df, window):
    """Swing high = harga tertinggi dalam jendela; swing low = terendah."""
    n = 2 * window + 1
    hi = df["high"].rolling(n, center=True).max()
    lo = df["low"].rolling(n, center=True).min()
    swing_high = df.loc[df["high"] == hi, ["date", "high"]].rename(columns={"high": "harga"})
    swing_low = df.loc[df["low"] == lo, ["date", "low"]].rename(columns={"low": "harga"})
    return pd.concat([swing_high, swing_low]).sort_values("harga").reset_index(drop=True)


def kelompokkan(titik, toleransi):
    """Gabungkan titik harga yang berdekatan menjadi zona (cluster)."""
    zona, grup = [], []
    for _, r in titik.iterrows():
        if grup and r["harga"] > (sum(g["harga"] for g in grup) / len(grup)) * (1 + toleransi):
            zona.append(grup)
            grup = []
        grup.append(r)
    if grup:
        zona.append(grup)
    hasil = []
    for g in zona:
        harga = [x["harga"] for x in g]
        hasil.append({
            "level": round(sum(harga) / len(harga), 2),
            "zona_bawah": round(min(harga), 2),
            "zona_atas": round(max(harga), 2),
            "sentuhan": len(g),
            "terakhir_tersentuh": max(x["date"] for x in g).date(),
        })
    return pd.DataFrame(hasil)


def pivot_klasik(df):
    """Pivot point dari candle terakhir (acuan harian)."""
    h, l, c = df.iloc[-1][["high", "low", "close"]]
    p = (h + l + c) / 3
    return {"R2": p + (h - l), "R1": 2 * p - l, "Pivot": p, "S1": 2 * p - h, "S2": p - (h - l)}


def analisis_tren(df):
    """Tentukan tren NAIK / TURUN / SIDEWAYS dari beberapa indikator (skor +1 / 0 / -1)."""
    c = df["close"]
    harga = c.iloc[-1]
    ma = {n: c.rolling(n).mean() for n in (MA_PENDEK, MA_MENENGAH, MA_PANJANG)}
    p, m, l = (ma[n].iloc[-1] for n in (MA_PENDEK, MA_MENENGAH, MA_PANJANG))
    rincian = []

    def tambah(nama, skor, ket):
        rincian.append({"indikator": nama, "skor": skor, "keterangan": ket})

    if pd.notna(p):
        tambah(f"Harga vs MA{MA_PENDEK}", 1 if harga > p else -1, f"{harga:,.2f} vs {p:,.2f}")
    if pd.notna(m):
        tambah(f"Harga vs MA{MA_MENENGAH}", 1 if harga > m else -1, f"{harga:,.2f} vs {m:,.2f}")
    if pd.notna(p) and pd.notna(m):
        tambah(f"MA{MA_PENDEK} vs MA{MA_MENENGAH}", 1 if p > m else -1, f"{p:,.2f} vs {m:,.2f}")
    if pd.notna(m) and pd.notna(l):
        tambah(f"MA{MA_MENENGAH} vs MA{MA_PANJANG}", 1 if m > l else -1, f"{m:,.2f} vs {l:,.2f}")

    # Arah kemiringan MA pendek (5 hari terakhir)
    if len(c) > MA_PENDEK + 5 and pd.notna(ma[MA_PENDEK].iloc[-6]):
        selisih = p - ma[MA_PENDEK].iloc[-6]
        tambah(f"Arah MA{MA_PENDEK}", 1 if selisih > 0 else -1, "menanjak" if selisih > 0 else "menurun")

    # Struktur pasar: dua swing high & swing low terakhir
    n = 2 * WINDOW + 1
    sh = df.loc[df["high"] == df["high"].rolling(n, center=True).max(), "high"].tail(2).tolist()
    sl = df.loc[df["low"] == df["low"].rolling(n, center=True).min(), "low"].tail(2).tolist()
    if len(sh) == 2 and len(sl) == 2:
        hh, hl = sh[1] > sh[0], sl[1] > sl[0]
        if hh and hl:
            tambah("Struktur harga", 1, "Higher High & Higher Low")
        elif not hh and not hl:
            tambah("Struktur harga", -1, "Lower High & Lower Low")
        else:
            tambah("Struktur harga", 0, "Campuran (belum jelas)")

    # Perubahan harga 20 hari terakhir
    if len(c) > 20:
        r = harga / c.iloc[-21] - 1
        tambah("Perubahan 20 hari", 1 if r > BATAS_RETURN else -1 if r < -BATAS_RETURN else 0, f"{r:+.2%}")

    total, maks = sum(x["skor"] for x in rincian), len(rincian)
    rasio = total / maks if maks else 0
    if rasio >= 0.4:
        label = "TREN NAIK (uptrend)"
    elif rasio <= -0.4:
        label = "TREN TURUN (downtrend)"
    else:
        label = "SIDEWAYS / belum jelas"
    return {"label": label, "skor": total, "maks": maks, "rincian": pd.DataFrame(rincian)}


def hitung_indikator(df):
    """Tambahkan kolom RSI, MACD, Bollinger Band, dan Volume/OBV ke data harian."""
    d = df.copy()
    c = d["close"]

    # RSI (metode Wilder)
    selisih = c.diff()
    rata_naik = selisih.clip(lower=0).ewm(alpha=1 / RSI_PERIODE, adjust=False, min_periods=RSI_PERIODE).mean()
    rata_turun = (-selisih.clip(upper=0)).ewm(alpha=1 / RSI_PERIODE, adjust=False, min_periods=RSI_PERIODE).mean()
    d["rsi"] = 100 - 100 / (1 + rata_naik / rata_turun)

    # MACD
    d["macd"] = c.ewm(span=MACD_CEPAT, adjust=False).mean() - c.ewm(span=MACD_LAMBAT, adjust=False).mean()
    d["macd_sinyal"] = d["macd"].ewm(span=MACD_SINYAL, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sinyal"]

    # Bollinger Band + %B (0 = di band bawah, 1 = di band atas)
    tengah = c.rolling(BB_PERIODE).mean()
    deviasi = c.rolling(BB_PERIODE).std(ddof=0)
    d["bb_tengah"] = tengah
    d["bb_atas"] = tengah + BB_STD * deviasi
    d["bb_bawah"] = tengah - BB_STD * deviasi
    d["bb_pctb"] = (c - d["bb_bawah"]) / (d["bb_atas"] - d["bb_bawah"])

    # Volume & On-Balance Volume (OBV)
    if "volume" in d.columns and d["volume"].fillna(0).sum() > 0:
        d["vol_rata"] = d["volume"].rolling(VOL_PERIODE).mean()
        d["obv"] = (np.sign(c.diff()).fillna(0) * d["volume"]).cumsum()
        d["obv_rata"] = d["obv"].rolling(VOL_PERIODE).mean()
    return d


def analisis_indikator(d):
    """Beri skor +1 (bullish) / 0 / -1 (bearish) untuk RSI, MACD, Bollinger, dan Volume."""
    x = d.iloc[-1]
    rincian = []

    def tambah(nama, skor, ket):
        rincian.append({"indikator": nama, "skor": skor, "keterangan": ket})

    # RSI
    if pd.notna(x["rsi"]):
        r = x["rsi"]
        if r >= 70:
            skor, ket = -1, "jenuh beli (rawan koreksi)"
        elif r <= 30:
            skor, ket = 1, "jenuh jual (peluang rebound)"
        elif r >= 50:
            skor, ket = 1, "momentum positif"
        else:
            skor, ket = -1, "momentum lemah"
        tambah(f"RSI({RSI_PERIODE})", skor, f"{r:.1f} - {ket}")

    # MACD
    if len(d) > MACD_LAMBAT + MACD_SINYAL:
        tanda = np.sign(d["macd_hist"].tail(4))
        silang = ""
        if tanda.iloc[-1] > 0 and (tanda.iloc[:-1] < 0).any():
            silang = " | baru golden cross"
        elif tanda.iloc[-1] < 0 and (tanda.iloc[:-1] > 0).any():
            silang = " | baru death cross"
        tambah("MACD vs Signal", 1 if x["macd_hist"] > 0 else -1,
               f"MACD {x['macd']:.2f} vs Signal {x['macd_sinyal']:.2f}{silang}")
        tambah("MACD vs garis nol", 1 if x["macd"] > 0 else -1,
               "di atas nol (momentum naik)" if x["macd"] > 0 else "di bawah nol (momentum turun)")

    # Posisi harga dalam Bollinger Band
    if pd.notna(x["bb_pctb"]) and np.isfinite(x["bb_pctb"]):
        pb = x["bb_pctb"]
        if pb >= 0.8:
            skor, ket = -1, "dekat/di atas band atas (rawan koreksi)"
        elif pb <= 0.2:
            skor, ket = 1, "dekat/di bawah band bawah (peluang rebound)"
        else:
            skor, ket = 0, "di area tengah band"
        tambah("Posisi Bollinger (%B)", skor,
               f"{pb:.2f} - {ket} | band {x['bb_bawah']:,.2f} s/d {x['bb_atas']:,.2f}")

    # Volume dan OBV
    if "vol_rata" in d.columns and pd.notna(x["vol_rata"]) and x["vol_rata"] > 0:
        rasio = x["volume"] / x["vol_rata"]
        arah = d["close"].iloc[-1] - d["close"].iloc[-2]
        if rasio >= VOL_LONJAKAN and arah > 0:
            skor, ket = 1, "naik dengan volume besar"
        elif rasio >= VOL_LONJAKAN and arah < 0:
            skor, ket = -1, "turun dengan volume besar"
        elif rasio < 0.5:
            skor, ket = 0, "volume sepi"
        else:
            skor, ket = 0, "volume normal"
        tambah(f"Volume vs rata-rata {VOL_PERIODE} hari", skor, f"{rasio:.2f}x - {ket}")
        if pd.notna(x["obv_rata"]):
            naik = x["obv"] > x["obv_rata"]
            tambah("OBV", 1 if naik else -1, "akumulasi (di atas rata-rata)" if naik else "distribusi (di bawah rata-rata)")

    total, maks = sum(i["skor"] for i in rincian), len(rincian)
    rasio_skor = total / maks if maks else 0
    if rasio_skor >= 0.4:
        label = "BULLISH"
    elif rasio_skor <= -0.4:
        label = "BEARISH"
    else:
        label = "NETRAL"
    return {"label": label, "skor": total, "maks": maks, "rincian": pd.DataFrame(rincian)}


