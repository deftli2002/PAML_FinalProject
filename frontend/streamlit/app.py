import json
from pathlib import Path

import streamlit as st
from detail import show_detail

st.set_page_config(page_title="RegLens", layout="wide")
DATA = Path(__file__).resolve().parent.parent / "data" / "products_enriched.json"
PAGE=25
@st.cache_data(show_spinner=False)
def load_products():
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    return raw["products"]
def dedupe(rows):
    seen,out=set(),[]
    for p in rows:
        s = float(p.get("risk_score") or 0)
        k=(
            str(p.get("brand_name","")).lower().strip(),
            str(p.get("generic_name","")).lower().strip(),
            s,)
        if k in seen:
            continue
        seen.add(k); out.append(p)
    return out
ss=st.session_state
ss.setdefault("selected_product_id", None)
ss.setdefault("list_page",0)
ss.setdefault("list_query","")
rows=load_products()
sel=next((p for p in rows if p["id"]==ss["selected_product_id"]), None)
if sel:
    show_detail(sel)
else:
    st.title("RegLens")
    q=st.text_input("Search","").lower().strip()
    r=[
        p
        for p in rows
        if not q
        or q
        in f"{p.get('brand_name','')} {p.get('generic_name','')} {p.get('entity_key','')}".lower()
    ]
    r.sort(key=lambda p: float(p.get("risk_score") or 0), reverse=True)
    r=dedupe(r)
    if q != ss["list_query"]:
        ss["list_query"], ss["list_page"] = q,0
    npg=max(1,(len(r)+PAGE-1)//PAGE)
    ss["list_page"]=min(ss["list_page"], npg - 1)
    pg=ss["list_page"]
    chunk=r[pg * PAGE : (pg + 1) * PAGE]

    a,b,c=st.columns(3)
    a.metric("items", len(r))
    b.metric("high", sum(1 for x in r if x.get("predicted_risk_level") == "High"))
    c.metric("page", f"{pg+1}/{npg}")
    p0,p1,_=st.columns([1,1,6])
    if p0.button("Prev", disabled=pg <= 0):
        ss["list_page"] -= 1
        st.rerun()
    if p1.button("Next", disabled=pg >= npg - 1):
        ss["list_page"]+=1
        st.rerun()

    for p in chunk:
        L,R=st.columns([5,1])
        with L:
            st.subheader(p["brand_name"])
            pct=int(float(p["risk_score"]) * 100)
            st.write(f"Generic: {p['generic_name']}")
            st.write(f"Risk: {p['predicted_risk_level']}")
            st.write(f"Action: {p['predicted_action']}")
            st.write(f"Score: {p['risk_score']}")
            st.write(f"Recall rate: {pct}%")
        with R:
            if st.button("Detail", key=p["id"]):
                ss["selected_product_id"]=p["id"]
                st.rerun()
        st.divider()
