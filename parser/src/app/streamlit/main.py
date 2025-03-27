import streamlit as st
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('corpus')
# from streamlit_pdf_viewer import pdf_viewer
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from pathlib import Path
import time
# from transformers import pipeline
import requests
import json
import logging
import yaml

with open(os.getcwd() + '/src/app/config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

print(config)
# ner_pipeline = pipeline("ner", model="yqelz/xml-roberta-large-ner-russian")
st.header("Проверка описи")
# st.text('Подрузить документа на проверку')
# st.button("Загрузить документ")
uploaded_file = st.file_uploader("Загрузить документ на проверку", type=["pdf","txt","docx"],
                                 help = "Выберите 'pdf' файл для отправки на проверку")
root = Path(os.getcwd())
#python -m streamlit run ./src/app/streamlit/main.py 

if 'click' not in st.session_state:
    st.session_state['click'] = 0

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    url = config['fastapi']
    #set timeout 
    ans = requests.post(url = f"{url}/check_pdf/",files =  {'file': bytes_data},
                            params={'query_id':1, "token":"y8f8TElxSOWCB1eZOAm0r0w66P29WT6m"},timeout=360)
    print(ans.content)
    data_df = pd.DataFrame(json.loads(ans.content)).T.iloc[:-1,:]
    st.data_editor(
        data_df,
        column_config={

            "Название": st.column_config.Column(
                "Название",
                help="Название проверки",
                width="large",
                disabled=True,
                required=True,
            ),
                "Результат": st.column_config.Column(
                "Результат",
                help="Streamlit **widget** commands 🎈",
                width="medium",
                disabled = True,
                required=True,
            )
        },
        hide_index=True,
        num_rows="fixed",
        # selection_mode = "multi-row",
        height = 1000,
    )

pg = st.navigation([
                    # st.Page(root / "src/app/api/web_gui.py",title="Конвертировать справочники"),
                    st.Page(root / "src/app/api/web_update1.py", title="Получить шаблон справочника")])
pg.run()
