"""Web app analisis saham. Jalankan: streamlit run app.py"""
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analisis as a

st.set_page_config(page_title="Analisis Saham: Support, Resisten & Tren", layout="wide")
st.title("📈 Analisis Support, Resisten & Tren Saham")
st.caption("Data harian dari Yahoo Finance. Saham BEI gunakan akhiran .JK (contoh: BBCA.JK). "
           "Hanya alat bantu analisis, bukan rekomendasi beli/jual.")

with st.sidebar:
    ticker = st.text_input("Ticker", "BBCA.JK").strip().upper()
    periode = st.selectbox("Periode data", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    with st.expander("Parameter lanjutan"):
        a.WINDOW = st.slider("Window swing (hari)", 2, 10, 5)
        a.TOLERANSI = st.slider("Toleransi zona (%)", 0.5, 5.0, 1.5, 0.5) / 100
        a.MIN_SENTUHAN = st.slider("Minimal sentuhan", 1, 5, 2)
        a.STOP_LOSS_PCT = st.slider("Stop loss di bawah support (%)", 1.0, 10.0, 2.0, 0.5) / 100


@st.cache_data(ttl=900, show_spinner="Mengambil data Yahoo Finance...")
def muat(t, p):
    return a.ambil_data(t, p)


@st.cache_data(ttl=3600, show_spinner="Mengambil data fundamental...")
def muat_fundamental(t):
    return a.ambil_fundamental(t)


if not ticker:
    st.info("Masukkan ticker di sidebar.")
    st.stop()
try:
    df = muat(ticker, periode)
except Exception as e:
    st.error(f"Gagal mengambil data: {e}")
    st.stop()

d = a.hitung_indikator(df)
harga = df["close"].iloc[-1]
level = a.kelompokkan(a.cari_swing(df, a.WINDOW), a.TOLERANSI)
if level.empty:
    level = pd.DataFrame(columns=["level", "zona_bawah", "zona_atas", "sentuhan", "terakhir_tersentuh"])
level = level[level["sentuhan"] >= a.MIN_SENTUHAN]
support = level[level["level"] < harga].sort_values("level", ascending=False).head(a.JUMLAH_LEVEL)
resisten = level[level["level"] > harga].sort_values("level").head(a.JUMLAH_LEVEL)
pivot = a.pivot_klasik(df)
tren = a.analisis_tren(df)
ind = a.analisis_indikator(d)
fund_data = muat_fundamental(ticker)
fund = a.analisis_fundamental(fund_data)

# ---- Identitas saham yang sedang dianalisis ----
nama = fund_data.get("nama") or ticker
st.header(f"{ticker} — {nama}")
info_tambahan = " • ".join(x for x in (fund_data.get("sektor"), fund_data.get("industri")) if x)
if info_tambahan:
    st.caption(info_tambahan)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Penutupan {df['date'].iloc[-1].date()}", f"{harga:,.2f}")
c2.metric("Tren", tren["label"].split(" (")[0], f"skor {tren['skor']:+d}/{tren['maks']}", delta_color="off")
c3.metric("Indikator", ind["label"], f"skor {ind['skor']:+d}/{ind['maks']}", delta_color="off")
if len(support) and len(resisten):
    s, r = support.iloc[0]["level"], resisten.iloc[0]["level"]
    sl = s * (1 - a.STOP_LOSS_PCT)
    rr = (r - harga) / (harga - sl) if harga > sl else float("nan")
    c4.metric("Support / Resisten terdekat", f"{s:,.2f} / {r:,.2f}", f"R:R {rr:.2f} | SL {sl:,.2f}", delta_color="off")
else:
    c4.metric("Support / Resisten terdekat", "-", "level valid belum ada", delta_color="off")

# ---- Ringkasan fundamental ----
mu = fund_data.get("mata_uang")
f1, f2, f3, f4, f5 = st.columns(5)
f1.metric("Market Cap", a.format_market_cap(fund_data.get("market_cap"), mu))
f2.metric("PER", f"{fund_data['per']:.2f}x" if a._angka(fund_data.get("per")) else "-")
f3.metric("PBV", f"{fund_data['pbv']:.2f}x" if a._angka(fund_data.get("pbv")) else "-")
f4.metric("ROE", f"{fund_data['roe']:.2%}" if a._angka(fund_data.get("roe")) else "-")
f5.metric("Fundamental", fund["label"].replace("FUNDAMENTAL ", "").title(),
          f"skor {fund['skor']:+d}/{fund['maks']}", delta_color="off")

# ---- Grafik ----
ada_vol = "volume" in d.columns and d["volume"].fillna(0).sum() > 0
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.52, 0.15, 0.18, 0.15],
                    subplot_titles=("Harga", f"RSI ({a.RSI_PERIODE})", "MACD", "Volume"))
x = d["date"]
fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Harga"), 1, 1)
for n, warna in ((a.MA_PENDEK, "orange"), (a.MA_MENENGAH, "royalblue")):
    fig.add_trace(go.Scatter(x=x, y=d["close"].rolling(n).mean(), name=f"MA{n}", line=dict(color=warna, width=1)), 1, 1)
for kol in ("bb_atas", "bb_bawah"):
    fig.add_trace(go.Scatter(x=x, y=d[kol], name=kol.replace("_", " ").title(),
                             line=dict(color="gray", width=1, dash="dot")), 1, 1)
for _, r_ in support.iterrows():
    fig.add_hline(y=r_["level"], line=dict(color="green", dash="dash"), annotation_text=f"S {r_['level']:,.2f}", row=1, col=1)
for _, r_ in resisten.iterrows():
    fig.add_hline(y=r_["level"], line=dict(color="red", dash="dash"), annotation_text=f"R {r_['level']:,.2f}", row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=d["rsi"], name="RSI", line=dict(color="purple")), 2, 1)
fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=2, col=1)
fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=2, col=1)
fig.add_trace(go.Bar(x=x, y=d["macd_hist"], name="Histogram", marker_color="lightgray"), 3, 1)
fig.add_trace(go.Scatter(x=x, y=d["macd"], name="MACD", line=dict(color="blue", width=1)), 3, 1)
fig.add_trace(go.Scatter(x=x, y=d["macd_sinyal"], name="Signal", line=dict(color="orange", width=1)), 3, 1)
if ada_vol:
    fig.add_trace(go.Bar(x=x, y=d["volume"], name="Volume", marker_color="steelblue"), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=d["vol_rata"], name=f"Rata-rata {a.VOL_PERIODE}h", line=dict(color="black", width=1)), 4, 1)
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04), margin=dict(t=60))
st.plotly_chart(fig, use_container_width=True)

# ---- Tabel ----
t1, t2, t3, t4, t5 = st.tabs(["Support & Resisten", "Tren", "Indikator", "Pivot Point", "Fundamental"])
with t1:
    k1, k2 = st.columns(2)
    k1.subheader("Support (area pertimbangan beli)")
    k1.dataframe(support, hide_index=True, use_container_width=True)
    k2.subheader("Resisten (area pertimbangan jual)")
    k2.dataframe(resisten, hide_index=True, use_container_width=True)
with t2:
    st.subheader(tren["label"])
    st.dataframe(tren["rincian"], hide_index=True, use_container_width=True)
with t3:
    st.subheader(ind["label"])
    st.dataframe(ind["rincian"], hide_index=True, use_container_width=True)
with t4:
    st.dataframe(pd.DataFrame(pivot.items(), columns=["Level", "Harga"]).round(2), hide_index=True)
with t5:
    st.subheader(fund["label"])
    st.dataframe(fund["rincian"], hide_index=True, use_container_width=True)
    st.caption("Ambang: PER < 15x murah, > 25x mahal | PBV < 1x murah, > 3x mahal | ROE >= 15% bagus, < 8% lemah. "
               "Sesuaikan dengan sektor (mis. bank wajar ber-PBV tinggi). Data dari Yahoo Finance dan bisa tidak lengkap.")

# ---- Unduh Excel ----
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    support.to_excel(w, sheet_name="Support", index=False)
    resisten.to_excel(w, sheet_name="Resisten", index=False)
    tren["rincian"].assign(kesimpulan=tren["label"]).to_excel(w, sheet_name="Tren", index=False)
    ind["rincian"].assign(kesimpulan=ind["label"]).to_excel(w, sheet_name="Indikator", index=False)
    fund["rincian"].assign(kesimpulan=fund["label"]).to_excel(w, sheet_name="Fundamental", index=False)
    d.to_excel(w, sheet_name="Data Harian", index=False)
st.download_button("⬇️ Unduh hasil (Excel)", buf.getvalue(),
                   file_name=f"analisis_{ticker.replace('.', '_')}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
