import json
from pathlib import Path
import streamlit as st
from detail import show_detail

st.set_page_config(page_title="RegLens",layout="wide")
DATA_PATH =Path(__file__).parent.parent / "mock" / "products.json"
if "selected_product_id" not in st.session_state:
    st.session_state["selected_product_id"] = None
with open(DATA_PATH, "r", encoding="utf-8") as f:
    products = json.load(f)
selected_product =None
for product in products:
    if product["id"]== st.session_state["selected_product_id"]:
        selected_product = product
        break

if selected_product is None:
    st.title("RegLens")
    st.write("Streamlit for FDA drugs recall risk rate.")
    st.caption("Search for drugs you want to know. For example: FUROSEMIDE、ENDOCET、ARTHROTEC、0054-3294、Acetaminophen")
    query =st.text_input("Search..")
    results =[]
    for product in products:
        if query:
            text =" ".join([product["brand_name"],product["generic_name"],]).lower()
            if query.lower().strip() not in text:
                continue
        results.append(product)

    results.sort(key=lambda x: x["risk_score"],reverse=True)
    col1, col2,col3 =st.columns(3)
    col1.metric("Products",len(results))
    col2.metric("High risk",len([x for x in results if x["predicted_risk_level"] =="High"]))
    for product in results:
        recall_probability = int(float(product["risk_score"])*100)
        left,right =st.columns([5, 1])
        with left:
            st.subheader(product["brand_name"])
            st.write("Generic name:",product["generic_name"])
            st.write("Risk level:",product["predicted_risk_level"])
            st.write("predicted action:",product["predicted_action"])
            st.write("Risk score:",product["risk_score"])
            st.write("Recall probability:",f"{recall_probability}%")
        with right:
            if st.button("Detail", key=product["id"]):
                st.session_state["selected_product_id"] =product["id"]
                st.rerun()
        st.divider()
else:
    show_detail(selected_product)