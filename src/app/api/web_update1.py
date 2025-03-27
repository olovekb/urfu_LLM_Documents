import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import json
import os
import yaml
with open(os.getcwd() + '/src/app/config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

if "clicked" not in st.session_state:
    st.session_state["clicked"] = False

if "count" not in st.session_state:
    st.session_state['count']  = 0

url = config['fastapi']
print(url)
# sub_number = st.selectbox(
#     "Выберите подномер справочника (шаблон) (0,1,2)",
#     [x for x in range(0,3)],
#     index=None,
#     placeholder="Select contact method...",
# )


files = {i:f"file_{i}.xlsx" for i in range(22)}
# Отображение кнопок для скачивания


st.header("Справочники")

data_content = requests.get(url = f"{url}/summary/")
dict_stats = json.loads(data_content.content)

stats_str = """<div style="width: 700px;">
Информация о справочниках
<table>
	<tbody>
        <tr><th>Cправочник</th><th>Количество записей</th><th>Количество дубликатов</th><th>Информация</th></tr>
"""

for n in dict_stats:
    if n in {'СП7','СП8'}:    
        info_s = "Количество записей в \n"
        for s in dict_stats[n].items():
            if s[0] in ['len','dup']:
                continue
            info_s+=f"{s[0]}: {s[1]};\n"
        new_line = f"<tr><td>{n}</td><td>{dict_stats[n]['len']}</td><td>{dict_stats[n]['dup']}</td><td>{info_s}</td></tr>"

    else:
        new_line = f"<tr><td>{n}</td><td>{dict_stats[n]['len']}</td><td>{dict_stats[n]['dup']}</td><td></td></tr>"
    stats_str+=new_line


stats_str+="""</tbody>
</table>
</div>
"""

st.markdown(
    f"""
    <div style="width: 700px;">
        <p>{stats_str}</p>
    </div>
    """,
    unsafe_allow_html=True
)

sp_help_info = """
<strong>Справочник №7</strong>  включает справочники с <strong>СП7</strong>  по <strong>СП14</strong>. 
Справочник представлен в виде xlsx таблицы содержащей записи "Заголовок дела" и "номер_справочника"<br>
Номера справочников для СП7 по СП14 заданы так: <br>
 7: 'Перечень документов постоянного хранения, подлежащих включению в опись Администраций сельских поселений Челябинской области'<br>
 8: 'Перечень документов постоянного хранения, подлежащих включению в опись №1 дел постоянного хранения Управления культуры администрации города/района'<br>
 9: 'Перечень документов постоянного хранения, подлежащих включению в опись №1 дел постоянного хранения Управления образования администрации города/района'<br>
10: 'Перечень документов постоянного хранения, подлежащих включению в опись №1 дел постоянного хранения государственного бюджетного учреждения здравоохранения «Районная больница г.__»'<br>
11: 'Перечень документов постоянного хранения, подлежащих включению в опись №1 дел постоянного хранения управлений / отделов имущества администрации муниципального образования'<br>
12: 'Перечень документов, подлежащих включению в опись №1 дел постоянного хранения Управления/Отдела по сельскому хозяйству администрации района'<br>
13: 'Перечень документов постоянного хранения, подлежащих включению в опись дел постоянного хранения комитетов/управлений/отделов администраций муниципальных образований, осуществляющих экономическую деятельность'<br>
14: 'Перечень документов, подлежащих включению в опись дел постоянного хранения управлений по физической культуре и спорту города/округа/района'<br>
<strong>Справочник №8</strong>  включает справочники с <strong>СП15</strong> по <strong>СП21</strong>. 
Номера справочников для <strong>СП15</strong>  по <strong>СП21</strong>  заданы так: <br>
15: Перечень документов территориальных и участковых избирательных комиссий, связанных с подготовкой и проведением общероссийского голосования по вопросу одобрения изменений в Конституцию ... <br>
16: Перечень документов территориальных и участковых избирательных комиссий, связанных с подготовкой и проведением выборов депутатов Государственной Думы Федерального Собрания ... <br>
17: Перечень документов территориальных и участковых избирательных комиссий, связанных с подготовкой и проведением выборов Губернатора Челябинской области ... <br>
18: Перечень документов окружных, территориальных и участковых избирательных комиссий, связанных с подготовкой и проведением выборов депутатов Законодательного Собрания Челябинской области седьмого созыва ... <br>
19: Перечень документов территориальных и участковых избирательных комиссий, связанных с подготовкой и проведением выборов Президента Российской Федерации ... <br>
20: Перечень документов избирательных комиссий, связанных с подготовкой и проведением выборов представительных органов муниципальных образований в Челябинской области ... <br>
21: Перечень документов избирательных комиссий, связанных с подготовкой и проведением выборов глав муниципальных образований Челябинской области ... <br>
"""

st.markdown(
    f"""
    <div style="width: 700px;">
        <p>{sp_help_info}</p>
    </div>
    """,
    unsafe_allow_html=True
)




sp_names = ["SP1","SP2","SP34","SP5","SP5LIST","SP7","SP8",
            "SP6LIST","SP61","SP62","SP63", "SP64","SP65","SP66","SP67", "SP68","SP69", "SP610", "SP611", "SP612"]
st.header("Получить cправочник")
sp_name_guide = st.selectbox(
    "Выберите номер справочника",
    [x for x in sp_names],
    index=None,
    placeholder="Select contact method...",
)

# sub_number_guide = st.selectbox(
#     "Выберите подномер справочника (0,1,2)",
#     [x for x in range(0,3)],
#     index=None,
#     placeholder="Select contact method...",
# )

if sp_name_guide is not None:
    data_guide = requests.get(url = f"{url}/guide/",params={"spname":sp_name_guide})
else:
    data_guide = None
if data_guide is not None:
    # Преобразуем данные в DataFrame для отображения
    # source = pd.read_csv(StringIO(data_content.text), sep = ';',encoding="cp1251")

    stream = BytesIO(data_guide.content)
    source = pd.read_excel(stream, engine='openpyxl')    
    source.to_excel('temp.xlsx', index=False)
    # data = source.to_csv(index=False).encode("utf-8")
    # file_name = files[number_guide]

    st.subheader(f"Файл справочник")
    st.dataframe(source)
    # Кнопка для скачивания
    with open("temp.xlsx", "rb") as template_file:
        template_byte = template_file.read()
    st.download_button(
        label=f"Скачать cправочник ",
        data=template_byte,
        file_name="СП.xlsx",
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )



st.header("Получить шаблон справочника")
pattern_info = """Шаблон справочника определяет название и порядок колонок справочника в системе. <br>
        Для добавления новых записей в справочник необходимо заполнить xlsx шаблон и <br>
        подрузить с помощью выпадающего списка 'Добавить строки в номер справочника'"""

st.markdown(
    f"""
    <div style="width: 700px;">
        <p>{pattern_info}</p>
    </div>
    """,
    unsafe_allow_html=True
)

sp_name = st.selectbox(
    "Выберите номер справочника (шаблон)",
    [x for x in sp_names],
    index=None,
    placeholder="Select contact method...",
)
if sp_name is not None:
    data_content = requests.get(url = f"{url}/template/",params={'spname':str(sp_name)})
else:
    data_content = None
if data_content:
    # Преобразуем данные в DataFrame для отображения
    # source = pd.read_csv(StringIO(data_content.text), sep = ';',encoding="cp1251")
    stream = BytesIO(data_content.content)
    source = pd.read_excel(stream, engine='openpyxl')    
    source.to_excel('temp.xlsx', index=False)
    # data = source.to_csv(index=False).encode("utf-8")
    file_name = "SP.xlsx"


    st.subheader(file_name)
    st.dataframe(source)
    # Кнопка для скачивания
    with open("temp.xlsx", "rb") as template_file:
        template_byte = template_file.read()
    st.download_button(
        label=f"Скачать {file_name}",
        data=template_byte,
        file_name=file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

st.header("Добавление новых записей в справочник")
st.caption("Впишите новые строки в файл шаблон-справочника.xlsx и подгрузите в систему")
file_new_rows = st.file_uploader("Excel файл с новыми строками", type=["xlsx"],
                                 help = "Загрузите xlsx файл")

append_info = """После выбора справочника из выпадаеюшего списка ниже будет произведена запись из файла xlsx в выбранных справочник""" 

st.markdown(
    f"""
    <div style="width: 700px;">
        <p>{append_info}</p>
    </div>
    """,
    unsafe_allow_html=True
)


sp_name_add = st.selectbox(
    "Добавить строки в номер справочника",
    [x for x in sp_names],
    index=None,
    placeholder="Select contact method...",
)
# sub_number_guide_add = st.selectbox(
#     "Добавить строки в подномер справочника (0,1,2,..12).",
#     [x for x in range(0,13)],
#     index=None,
#     placeholder="Select contact method...",
# )

if file_new_rows is not None:
    bytes_data = file_new_rows.read()
    # print(number_guide_add, sub_number_guide_add)
    if sp_name_add in sp_names:
        ans = requests.post(url = f"{url}/append/",files =  {'file': bytes_data},
                            params={'spname':sp_name_add})
        st.header(f"Результат добавления в sp{sp_name_add}:{ans.content}")


st.header("Обновить справочник (удаляет все записи и вписывает новые из файла)")

file_new = st.file_uploader("Excel файл для обновления", type=["xlsx"],
                                 help = "Загрузите xlsx файл")


sp_name_new= st.selectbox(
    "Обновить справочник",
    [x for x in sp_names],
    index=None,
    placeholder="Select contact method...",
)
# sub_number_guide_add = st.selectbox(
#     "Добавить строки в подномер справочника (0,1,2,..12).",
#     [x for x in range(0,13)],
#     index=None,
#     placeholder="Select contact method...",
# )

if file_new is not None:
    bytes_data = file_new.read()
    # print(number_guide_add, sub_number_guide_add)
    if sp_name_new in sp_names:
        ans = requests.post(url = f"{url}/upload/",files =  {'file': bytes_data},
                            params={'spname':sp_name_new})
        st.header(f"Результат добавления в sp{sp_name_new}:{ans.content}")


st.markdown(
    f"""
    <div style="width: 700px;">
        <p>Версия 0.1.1</p>
    </div>
    """,
    unsafe_allow_html=True
)

# st.header("Восстановление справочников")
# ans = requests.get(url = f"{url}/read_backup/")
# ans_dict = json.loads(ans.text)
# option = st.selectbox(
#     "Выберите папку с backup для восстановления",
#     ans_dict
# )




# print('start')
# def click_restore(option):
#     data_content = requests.post(url = f"{url}/restore/", params={'dir_name':option})
#     res = json.loads(data_content.text)
#     st.header(f"Результат восстановления: {res['status']}")


# def click_test():
#     print("Нажали")
#     st.session_state['count'] +=1
#     st.session_state["clicked"] = True
    


# restore_btn = st.button(f'Восставновить справочники из резервной папки {option}', on_click=click_restore(option))
# restore_btn.sta
# st.header("Проверка справочника")
# test_btn = st.button(f"Проверка справочников", on_click=click_test())

# if st.session_state["clicked"]:
#     print("Запустили")
#     st.header(f"запустили {st.session_state['count']}", )
#     ans = requests.get(url = f"{url}/pdf_test")
#     d = json.loads(ans.content)
#     st.data_editor(pd.DataFrame(d).T)

