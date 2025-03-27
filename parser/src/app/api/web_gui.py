import streamlit as st
import requests
import json
import pandas as pd


URL = "http://212.193.88.193:8005/update_guide/"
uploaded_file = st.file_uploader("Загрузить справочник", type=["pdf","csv","xls","xlsx"],
                                 help = "Выберите 'pdf' файл справочника №:")

sp_names = ["SP1","SP2","SP34","SP5","SP5LIST","SP7","SP8"]
option = st.selectbox(
    "How would you like to be contacted?",
    ([f"{x}" for x in sp_names]),
)
st.write("Выберите номер справочника:", option)
sent_button = st.button(f' Обновить справочник №{option}')
if sent_button:
    if uploaded_file:
        bytes_data = uploaded_file
        r = requests.post(url = URL, files = {'file': bytes_data},data={'id':int(option)})
        ans = json.loads(r.content)
        st.text(f"{ans['id']}, f{ans['file']}")
buttons = {} 
txt_inputs = {}
for k in range(0,24):
    col_button, col_status = st.columns(2, vertical_alignment="bottom")
    # with col_name:
    #     st.text(f"№{k}")
    with col_button:
        if k == 0:
            placeholder = st.empty()
            placeholder.markdown("Справочник №")
        else:
            sent_button = st.button(f'Обновить справочник №{k}', key = f"bt_{k}")
            buttons[k] = sent_button
    with col_status:
        if k == 0:
            placeholder = st.empty()
            placeholder.markdown("Статус")

        placeholder = st.empty()
        placeholder.markdown("")
        txt_inputs[k] = placeholder
    #     # with st.form(key=f"st_{k}"):
    #         text_input = st.text_area("Статус:", key = "st_"+str(k))
    #         txt_inputs[k] = text_input  

data_df = pd.DataFrame([i for i in range(1,24)])
# with col_status:
    
#     st.data_editor(
#         data_df,
#         num_rows="fixed",
#         # selection_mode = "multi-row",
#         height = 1000,
#     )
# for k in buttons:
#     if buttons[k]:
#         txt_inputs[k].markdown(str(k))

for k in buttons:
    if buttons[k]:
        if uploaded_file:
            bytes_data = uploaded_file
            r = requests.post(url = URL, files = {'file': bytes_data},data={'id':int(k)})
            ans = json.loads(r.content)
            txt_inputs[k].markdown(ans['status'])