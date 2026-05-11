import pandas as pd
import streamlit as st


def _tbl(rows):
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )


def show_detail(p):
    if st.button("Back"):
        st.session_state["selected_product_id"]=None
        st.rerun()
    st.title(p["brand_name"])
    pct=int(float(p["risk_score"]) * 100)
    st.write(f"Generic: {p.get('generic_name')}")
    st.write(f"Risk: {p['predicted_risk_level']}")
    st.write(f"Action: {p['predicted_action']}")
    st.write(f"Score: {p['risk_score']}")
    st.write(f"Recall rate: {pct}%")
    st.write(f"Signal: {p.get('quality_signal')}")
    st.write(f"Summary: {p.get('summary')}")
    f = p["features"]
    with st.expander("features"):
        st.dataframe(
            pd.DataFrame([f]).T.rename(columns={0: "value"}),
            use_container_width=True,
        )
    st.write("Ingredients:")
    _tbl(p.get("active_ingredients"))
    st.write("Recall history:")
    _tbl(p.get("recall_history"))
