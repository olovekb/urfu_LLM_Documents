import streamlit as st

pg = st.navigation([st.Page("web_gui.py",title="Конвертировать справочники"),
                    st.Page("web_update1.py", title="Получить шаблон справочника")])
pg.run()