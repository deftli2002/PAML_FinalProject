import pandas as pd
import streamlit as st
def show_detail(product):
    recall_probability =int(float(product["risk_score"])*100)
    if st.button("go back"):
        st.session_state["selected_product_id"] =None
        st.rerun()
    st.title(product["brand_name"])
    st.write(product["generic_name"])
    st.write("Risk:",product["predicted_risk_level"])
    st.write("What may happen:",product["predicted_action"])
    st.write("Score:",product["risk_score"])
    st.write("Recall chance in like 12 months:", f"{recall_probability}%")
    st.write("Signal:", product["quality_signal"])
    st.write("Short summary thing:",product["summary"])
    st.write("Active ingredients stuff")
    st.dataframe(pd.DataFrame(product["active_ingredients"]),
        hide_index=True,
        use_container_width=True)
    st.write("Packaging info")
    st.dataframe(pd.DataFrame(product["packaging"]),
        hide_index=True,
        use_container_width=True)
    st.write("Old recall records")
    st.dataframe(pd.DataFrame(product["recall_history"]),
        hide_index=True,
        use_container_width=True)