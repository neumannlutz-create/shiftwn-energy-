"""
ShiftWN Energy – Backtest-Controlling für Energiemärkte
Phase 1: Rückschauende Analyse. Zeigt, wie ShiftWN auf historischen
Energiedaten eingegriffen hätte – Signale, Regime-Brüche, Widerspruch
gegen eine naive Strategie.
Start mit frei verfügbaren Energie-Instrumenten (Gas, Öl) über yfinance.
Echte Strom-Spotdaten (ENTSO-E Day-Ahead) werden später als Datenadapter
angebunden (Token erforderlich).
Patentierter geometrischer Kern: Triangle · Vortex · Impulse FFT · Photonics.
Patent EPA EP25221251.9 / SPECEPO-1/2.  Keine Anlageberatung.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="ShiftWN Energy", layout="wide", page_icon="⚡")

# ---------------- Look: warmes Hellgrau + Petrol (wie Haupt-App) ----------------
BG="#e9e7e2"; BG_ALT="#e3e1db"; CARD="#f1efea"; BORDER="#cfccc4"
TEAL="#1f6f6b"; TEXT="#2b2f36"; MUTED="#6b6f78"; FAINT="#969aa2"
BUY="#1f6f6b"; SELL="#b5524a"; HOLD="#8a8077"; SHOCK="#b07a2e"

st.markdown(f"""
<style>
    .stApp {{background:{BG};color:{TEXT};}}
    section[data-testid="stSidebar"]{{background:{BG_ALT};border-right:1px solid {BORDER};}}
    h1{{color:{TEXT};font-weight:700;letter-spacing:-.3px;margin-bottom:0;}}
    .app-sub {{color:{MUTED};}}
    h2{{color:{MUTED};font-weight:600;font-size:.95rem;text-transform:uppercase;
        letter-spacing:2px;border-bottom:1px solid {BORDER};padding-bottom:8px;margin-top:18px;}}
    h2 .num{{color:{TEAL};font-weight:700;margin-right:8px;}}
    .stMetric{{background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:16px 18px;
               box-shadow:0 1px 3px rgba(43,47,54,.06);}}
    [data-testid="stMetricValue"]{{color:{TEAL};font-size:1.35rem;font-weight:700;}}
    [data-testid="stMetricLabel"]{{color:{MUTED};font-size:.78rem;text-transform:uppercase;letter-spacing:.5px;}}
    .stButton>button{{background:{TEAL};color:#f4f3ef;font-weight:600;border:none;border-radius:8px;}}
    .block-container{{padding-top:2.4rem;max-width:1400px;}}
    .infobox{{background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:14px 18px;margin:6px 0 14px 0;}}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  PATENTIERTER SHIFTWN-KERN  (identisch zur Haupt-App)
# ============================================================
def _normalize(window):
    c=window[:,3]; ref=np.median(c) if np.median(c)>0 else 1.0
    scale=np.median(np.abs(np.diff(c))) or 0.01*ref
    n=np.zeros_like(window,dtype=float)
    n[:,0]=(window[:,0]-ref)/scale; n[:,1]=(window[:,1]-ref)/scale
    n[:,2]=(window[:,2]-ref)/scale; n[:,3]=(c-ref)/scale
    n[:,4]=window[:,4]/(np.median(window[:,4]) or 1.0); return n

def triangle(window):
    c=_normalize(window)[:,3]; x=np.arange(len(c))
    if len(c)<5: return {"score":0.0,"kappa":0.0,"dir":0.0}
    lin=np.polyfit(x,c,1); fit=np.polyval(lin,x); sst=np.sum((c-np.mean(c))**2)+1e-9
    r2=1-np.sum((c-fit)**2)/sst; h=len(c)//2; re_=np.ptp(c[:h]); rl=np.ptp(c[h:])
    wedge=np.clip((re_-rl)/(re_+1e-9),0,1)
    return {"score":float(np.clip(abs(lin[0])*6,0,1)),"kappa":float(np.clip(max(r2,wedge*0.9),0,1)),"dir":float(np.tanh(lin[0]*3))}

def vortex(window):
    c=_normalize(window)[:,3]
    if len(c)<6: return {"score":0.0,"kappa":0.0,"dir":0.0,"vol_break":0.0}
    pos=c-np.mean(c); vel=np.gradient(pos); ang=np.arctan2(vel,pos)
    dphi=np.diff(ang); dphi=(dphi+np.pi)%(2*np.pi)-np.pi
    score=float(np.clip(np.mean(np.abs(dphi))/(np.std(dphi)+1e-8),0,1))
    rot=np.clip(1.0-np.std(dphi)/(np.pi/2),0,1); h=len(c)//2
    v1=np.std(np.diff(c[:h]))+1e-9; v2=np.std(np.diff(c[h:]))+1e-9
    vb=float(np.clip(abs(np.log(v2/v1))/1.5,0,1))
    return {"score":score,"kappa":float(np.clip(rot*(1-vb),0,1)),"dir":float(np.tanh(np.polyfit(np.arange(len(c)),c,1)[0]*3)),"vol_break":vb}

def impulse(window):
    c=_normalize(window)[:,3]; r=np.diff(c)
    if len(r)<5: return {"score":0.0,"kappa":0.0,"dir":0.0}
    r=r-np.mean(r); w=np.hanning(len(r)); spec=np.abs(np.fft.rfft(r*w))**2
    ss=np.sum(spec) or 1.0; p=spec/ss; ent=-np.sum(p*np.log(p+1e-12))/np.log(len(p))
    return {"score":float(np.max(p)),"kappa":float(np.clip((1-ent)*1.8,0,1)),"dir":0.0}

def photonics(tri,vor,imp,prev_mk=None):
    ks=np.array([tri["kappa"],vor["kappa"],imp["kappa"]]); ss=np.array([tri["score"],vor["score"],imp["score"]])
    w=np.ones(3)/3 if ks.sum()<1e-6 else ks/ks.sum()
    mk=float(ks.mean()); drop=0.0 if prev_mk is None else max(0.0,prev_mk-mk); vb=vor.get("vol_break",0.0)
    rb=(vb>0.45 and drop>0.12)
    if rb: modus,dom,lead="REGIME-BRUCH / SCHOCK","Impulse FFT",2; drift=vor["dir"]
    else:
        lead=int(np.argmax(ks)); modus=["TRENDING","RANGING","VOLATILE"][lead]
        dom=["Triangle","Vortex","Impulse FFT"][lead]; drift=[tri["dir"],vor["dir"],vor["dir"]][lead]
    return {"w":w,"modus":modus,"dominant":dom,"mean_kappa":mk,"regime_break":rb,"drift":drift}

def _build(seg):
    n=len(seg); w=np.zeros((n,5)); w[:,3]=seg; w[:,0]=seg*0.97; w[:,1]=seg*1.08; w[:,2]=seg*0.92; w[:,4]=15000; return w

def analyse_at(closes, end, win=50):
    cur=closes[end-win:end]; prev=closes[end-2*win:end-win] if end>=2*win else None; pmk=None
    if prev is not None and len(prev)>=20:
        pmk=np.mean([triangle(_build(prev))["kappa"],vortex(_build(prev))["kappa"],impulse(_build(prev))["kappa"]])
    return photonics(triangle(_build(cur)),vortex(_build(cur)),impulse(_build(cur)),pmk)

def signal_of(ph, dth, kmin):
    if ph["regime_break"]: return "SCHOCK"
    if ph["mean_kappa"]>kmin and ph["drift"]>dth: return "BUY"
    if ph["mean_kappa"]>kmin and ph["drift"]<-dth: return "SELL"
    return "HOLD"

SIG_COL={"BUY":BUY,"SELL":SELL,"HOLD":HOLD,"SCHOCK":SHOCK}

# ============================================================
#  DATEN: Energie-Instrumente (frei) + Platzhalter für ENTSO-E
# ============================================================
ENERGIE={
    "TTF Gas (Europa)": "TTF=F",
    "Erdgas Henry Hub (US)": "NG=F",
    "Brent Öl": "BZ=F",
    "WTI Öl": "CL=F",
}
ZEIT={"2 Jahre (täglich)":("2y","1d"),"1 Jahr (täglich)":("1y","1d"),"6 Monate (täglich)":("6mo","1d")}

@st.cache_data(ttl=600, show_spinner=False)
def lade_energie(ticker, period, interval):
    try:
        df=yf.download(ticker,period=period,interval=interval,progress=False,auto_adjust=True,timeout=10)
    except Exception:
        df=None
    if df is not None and not df.empty:
        c=df["Close"].values.flatten()
        idx=df.index
        m=~np.isnan(c)
        return c[m], idx[m]
    return None, None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ShiftWN Energy")
    st.caption("Phase 1 · Backtest-Controlling")
    st.divider()
    markt_name=st.selectbox("Energie-Instrument", list(ENERGIE.keys()))
    zeit_name=st.selectbox("Zeitraum", list(ZEIT.keys()))
    st.divider()
    st.subheader("Signal-Schwellen")
    drift_th=st.slider("Drift (Minimum)",0.04,0.30,0.06,0.01)
    kappa_min=st.slider("Mindest-Aussagekraft",0.20,0.70,0.35,0.01)
    st.divider()
    st.subheader("Vergleichsannahme")
    annahme=st.selectbox("Naive Strategie, der ShiftWN gegenübergestellt wird",
                         ["Immer Long (Kauf-Bias)","Immer Short (Verkauf-Bias)"])
    st.divider()
    st.caption("Echte Strom-Spotdaten (ENTSO-E Day-Ahead) folgen als Datenadapter, sobald der API-Token vorliegt.")

ticker=ENERGIE[markt_name]; period,interval=ZEIT[zeit_name]

# ---------------- Kopf ----------------
st.title("⚡ ShiftWN Energy")
st.markdown(f"<div class='app-sub'>Backtest-Controlling für Energiemärkte · Patent EPA EP25221251.9 · "
            f"Stand {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</div>", unsafe_allow_html=True)

with st.spinner(f"Lade {markt_name} ..."):
    closes, idx = lade_energie(ticker, period, interval)

if closes is None or len(closes)<120:
    st.warning(f"Für {markt_name} konnten gerade nicht genügend Daten geladen werden. "
               f"Bitte anderes Instrument oder Zeitraum wählen – oder kurz erneut versuchen.")
    st.stop()

# ============================================================
#  BACKTEST RECHNEN
# ============================================================
win=50
results=[]  # (position, signal, modus, drift, kappa, preis)
for end in range(2*win, len(closes)):
    ph=analyse_at(closes, end, win)
    sig=signal_of(ph, drift_th, kappa_min)
    results.append((end, sig, ph["modus"], ph["drift"], ph["mean_kappa"], float(closes[end-1])))

if not results:
    st.warning("Zu wenig Datenpunkte für einen Backtest in diesem Zeitraum.")
    st.stop()

sigs=[r[1] for r in results]
from collections import Counter
cnt=Counter(sigs)
n=len(results)

# Widerspruch gegen die naive Annahme
if annahme.startswith("Immer Long"):
    widerspruch=[r for r in results if r[1] in ("SELL","SCHOCK")]
    annahme_kurz="Long"
else:
    widerspruch=[r for r in results if r[1] in ("BUY","SCHOCK")]
    annahme_kurz="Short"

# Eingriffe = Signalwechsel
wechsel=[(results[k][0],results[k-1][1],results[k][1],results[k][5]) for k in range(1,len(results)) if results[k][1]!=results[k-1][1]]
schocks=[r for r in results if r[1]=="SCHOCK"]

# ============================================================
#  1) ÜBERBLICK
# ============================================================
st.markdown(f"## <span class='num'>1</span> Überblick — {markt_name}", unsafe_allow_html=True)
m1,m2,m3,m4=st.columns(4)
m1.metric("Analysierte Tage", f"{n}")
m2.metric("Eingriffe (Signalwechsel)", f"{len(wechsel)}")
m3.metric(f"Widerspruch zu '{annahme_kurz}'", f"{len(widerspruch)/n*100:.0f}%")
m4.metric("Erkannte Regime-Brüche", f"{len(schocks)}")

st.markdown(f"<div class='infobox'>In diesem Zeitraum hätte ShiftWN <b>{len(wechsel)} Mal</b> das Signal "
            f"gewechselt und in <b>{len(widerspruch)} von {n} Fällen</b> einer reinen "
            f"<b>{annahme_kurz}-Annahme widersprochen</b>. Davon <b>{len(schocks)}</b> Regime-Brüche, "
            f"in denen die gewohnte Marktstruktur zusammenbrach.</div>", unsafe_allow_html=True)

# ============================================================
#  2) VERLAUF MIT SIGNALEN
# ============================================================
st.markdown(f"## <span class='num'>2</span> Kursverlauf mit ShiftWN-Eingriffen", unsafe_allow_html=True)

xs=list(range(len(closes)))
fig=go.Figure()
fig.add_trace(go.Scatter(y=closes, x=xs, mode="lines", line=dict(color=MUTED,width=1.5), name=markt_name))

# Signalpunkte einfärben
for sig,colr in [("BUY",BUY),("SELL",SELL),("SCHOCK",SHOCK)]:
    pts=[(r[0]-1, r[5]) for r in results if r[1]==sig]
    if pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
            marker=dict(color=colr,size=6,line=dict(width=0)), name=sig))
fig.update_layout(height=440, template="plotly_white", paper_bgcolor=BG, plot_bgcolor=CARD,
                  margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h",yanchor="bottom",y=1.0),
                  xaxis=dict(gridcolor=BORDER,title="Handelstage"), yaxis=dict(gridcolor=BORDER,title="Preis"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Grün = BUY-kohärent · Rot = SELL-kohärent · Bernstein = Regime-Bruch. "
           "Graue Linie = Kursverlauf. ShiftWN ist Signalgeber, keine Order-Ausführung.")

# ============================================================
#  3) SIGNAL-VERTEILUNG + EINGRIFFS-LISTE
# ============================================================
st.markdown(f"## <span class='num'>3</span> Auswertung", unsafe_allow_html=True)
cL,cR=st.columns([2,3])
with cL:
    st.markdown("**Signal-Verteilung**")
    figv=go.Figure(go.Bar(
        x=[cnt.get("BUY",0),cnt.get("HOLD",0),cnt.get("SELL",0),cnt.get("SCHOCK",0)],
        y=["BUY","HOLD","SELL","SCHOCK"], orientation="h",
        marker_color=[BUY,HOLD,SELL,SHOCK],
        text=[cnt.get("BUY",0),cnt.get("HOLD",0),cnt.get("SELL",0),cnt.get("SCHOCK",0)], textposition="auto"))
    figv.update_layout(height=240, template="plotly_white", paper_bgcolor=BG, plot_bgcolor=CARD,
                       margin=dict(l=0,r=10,t=4,b=20), xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
    st.plotly_chart(figv, use_container_width=True)
with cR:
    st.markdown("**Letzte Eingriffe (Signalwechsel)**")
    if not wechsel:
        st.caption("Keine Signalwechsel in diesem Zeitraum.")
    else:
        for pos,prev,now,preis in reversed(wechsel[-12:]):
            datum = str(idx[pos-1])[:10] if idx is not None and pos-1 < len(idx) else f"Tag {pos}"
            cnow=SIG_COL.get(now,MUTED)
            st.markdown(f"<div class='infobox' style='padding:8px 12px;margin-bottom:6px'>"
                        f"<span style='color:{MUTED};font-size:.78rem'>{datum}</span> · "
                        f"<span style='color:{MUTED}'>{prev}</span> → "
                        f"<span style='color:{cnow};font-weight:700'>{now}</span> "
                        f"<span style='color:{FAINT};font-size:.76rem'>· Preis {preis:,.2f}</span></div>",
                        unsafe_allow_html=True)

st.markdown("---")
st.caption("ShiftWN Energy · Phase 1: Backtest-Controlling · Patentierter geometrischer Kern "
           "(Triangle · Vortex · Impulse FFT · Photonics). Rückschauende Analyse auf historischen Daten. "
           "Vergangene Signale sind keine Garantie für künftige Ergebnisse. Keine Anlageberatung.")
