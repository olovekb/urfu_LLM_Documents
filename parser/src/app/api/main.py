from fastapi import FastAPI, File, UploadFile, Form, HTTPException, applications, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import pandas as pd
import os
import sys
import requests
from fastapi import FastAPI, Request, Response
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles


# from pydantic import AnyWebsocketUrl
# import fastapi_offline_swagger_ui

import uvicorn
from pathlib import Path


sys.path.append(os.getcwd())
# sys.path.append("C:\\Project\\documents\\")
import os

os.environ["TOKENIZERS_PARALLELISM"] = "True"

from src.app.api.utils import create_empty_ans
from src.app.checker.utils.text_utils import (
    read_text_documents,
    prep_dataset_sp5,
    prep_sp6_tables,
)
from src.app.checker.checker import (
    check_1,
    check_2,
    check_3,
    check_4,
    check_5,
    check_6,
    check_8,
    check_9,
    check_13,
    check_14,
    check_15_and_16,
    check_17,
    check_21,
    check_22,
    check_26,
    check_28,
)
from src.app.checker.checker import (
    check_0,
    check_7,
    check_10,
    check_11,
    check_12,
    check_18,
    check_19,
    check_20,
    check_23,
    check_25,
    check_27,
    check_2_search,
)
from src.app.checker.utils.table_utils import get_table_documents, drop_table_lines
from src.app.checker.checker import names_checks
from src.app.checker.utils.table_utils import get_tocken

from sp_preprocess import sp_converter
from src.app.api.utils import (
    SpDbManager,
    SpManager,
    path2directory,
    dump_pkl,
    allowed_file,
    bool2str,
    path2backup,
    make_backup,
    copy_pkl_files,
    path2save,
)
import logging
import warnings
import io
from transformers import pipeline
import json
import zipfile
from src.app.checker.utils.file_utils import log_execution
from src.app.api.dbase import DBManager
import asyncio
import yaml
import time
with open(os.getcwd() + "/src/app/config.yaml") as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

print(config)
if bool(config['local']):
    ner_pipeline = pipeline("ner", model="/root/huggingface/hub/models--yqelz--xml-roberta-large-ner-russian/snapshots/4e03a3a079f5bfdb678eb9ee01235e25a7aef5f6/")
else:
    ner_pipeline = pipeline("ner", model="yqelz/xml-roberta-large-ner-russian")


LOGGER = logging.getLogger("api_log")
LOGGER.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f"log_api.txt")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)

dbase = DBManager(path2directory() + "base.db")


def read_splist(dbmanager):
    # метод в классе sp_manager
    # path2catalog = path2directory()
    # sp1_name = "sp1.pkl"
    # sp2_name = "sp2.pkl"
    # sp3_name = "sp3.pkl"
    # sp4_name = "sp4.pkl"
    # sp5_name = "sp5.pkl"
    # sp5_list_name = "sp5_list.pkl"
    # sp6_list_name = "sp6_list.pkl"
    # sp5_list_prep_name = "sp5_list_prep.pkl" # можно получить запустив prep_sp6_lists (для выполнения нужен sp5_list.pkl)
    # sp6_list_prep_name = "sp6_list_prep.pkl" # можно получить запустив prep_sp6_lists (для выполнения нужен sp6_list.pkl)
    # sp6_all_tables_name = "sp6_all_tables.pkl" # можно получить запустив prep_sp6_tables
    # sp7_name = "sps7_14_merge.pkl"
    # sp8_name = "sps15_21_merge.pkl"

    sp1 = dbmanager.get_table("SP1")
    sp1 = sp1[["Наименование организации", "Архивный отдел"]]
    sp2 = dbmanager.get_table("SP2")
    sp34 = dbmanager.get_table("SP34")
    sp3 = sp34[["Архив", "Наименование"]]
    sp4 = sp34[["Наименование", "Архив", "Номер фонда"]]

    sp5 = dbmanager.get_table("SP5")
    sp5.iloc[:, 0] = sp5.iloc[:, 0].astype("str")
    sp5_list_prep = dbmanager.get_table("SP5LIST")
    sp6_list_prep = dbmanager.get_table("SP6LIST")
    sp6_list_prep.iloc[:, 0] = sp6_list_prep.iloc[:, 0].astype("str")

    sp6_all_tables = prep_sp6_tables(dbmanager)
    sp6_all_tables.iloc[:, 0] = sp6_all_tables.iloc[:, 0].astype("str")
    sp7 = dbmanager.get_table("SP7")
    sp8 = dbmanager.get_table("SP8")
    # print(sp1.shape, sp2.shape, sp34.shape, sp6_all_tables.shape, sp7.shape)
    sp7["tocken"] = sp7.iloc[:, 0].apply(lambda t: get_tocken(t, True))

    sps = {
        61: sp6_list_prep,
        62: sp6_all_tables,
        7: sp7,
        8: sp8,
    }

    return sp1, sp2, sp3, sp4, sp5, sp5_list_prep, sps

# @log_execution(LOGGER)
async def check_document(fpath: str, ner_pipeline, id):
    stime = time.time()
    sp1, sp2, sp3, sp4, sp5, sp5_list_prep, sps = read_splist(dbase)
    log = """Ошибка в проверке №{num}. Текст ошибки: "{err}". Необходимо проверить: 1)корректность входных данных. Опись парсится в переменные table_doc и clear_table в файле main.py; 2)инициализацию переменной res_check_{num} в файле main.py и корректность работы функции check_{num} в файле checker.py."""

    LOGGER.debug(fpath)
    dict_text, full_text, page_blocks = read_text_documents(fpath)
    #Если файл не pdf и не прочитал текст внутри
    if all(map(lambda x: x=='', [v for k,v in dict_text.items()])):
        ans = {}
        for k, v in names_checks.items():
            if k == 0:
                ans[k] = {"name": v, "status": "False", "coment": "Неудалось прочитать текст внутри документа."}
                continue
            ans[k] = {"name": v, "status": "False", "coment": ""}
        LOGGER.info(f"Время обработки для id = {id}, t = {(time.time() - stime):.2f} c")
        return ans
    try:
        table_doc, bad_columns = get_table_documents(fpath)
        clear_table = drop_table_lines(table_doc)
    except:
        table_doc, clear_table = None, None

    res_check_0 = False
    comment_0 = ""
    try:
        res_check_0, comment_0 = check_0(fpath)
    except Exception as err:
        res_check_0 = False
        comment_0 = f"Не удалось проверить корректность таблицы в  файле:{fpath.name} "

    if res_check_0 == False:
        ans = create_empty_ans(fpath.name)
        LOGGER.info(f"Время обработки для id = {id}, t = {(time.time() - stime):.2f} c")
        return ans
    try:
        res_check_1, comment_1 = check_1(dict_text, sp2)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=1))
        res_check_1 = "Не удалось"
        comment_1 = ""
    try:
        res_check_2, comment_2 = check_2(dict_text, sp1, sp3)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=2))
        res_check_2 = "Не удалось"
        comment_2 = ""
    try:
        res_check_3 = check_3(dict_text, page_blocks, sp4)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=3))
        res_check_3 = "Не удалось"
    try:
        res_check_4, comment_4 = check_4(dict_text, page_blocks)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=4))
        res_check_4 = "Не удалось"
        comment_4 = ""
    try:
        res_check_5 = check_5(dict_text)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=5))
        res_check_5 = "Не удалось"
    try:
        res_check_6 = check_6(dict_text)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=6))
        res_check_6 = "Не удалось"
    try:
        res_check_7 = check_7(dict_text, clear_table)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=7))
        res_check_7 = "Не удалось"

    try:
        res_check_8, comment_8 = check_8(dict_text)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=8))
        res_check_8 = "Не удалось"
        comment_8 = ""

    try:
        res_check_9 = check_9(dict_text)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=9))
        res_check_9 = "Не удалось"

    if res_check_0:  # проверки по таблице
        try:
            res_check_10, comment_10 = check_10(
                table_doc, sp5_list_prep, dict_text, sp5
            )
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=10))
            res_check_10 = "Не удалось"
            comment_10 = ""

        try:
            res_check_11, comment_11 = check_11(dict_text, clear_table)
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=11))
            res_check_11 = "Не удалось"
            comment_11 = ""

        try:
            res_check_12, comment_12 = check_12(dict_text, clear_table, sps[7])
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=12))
            res_check_12 = "Не удалось"
            comment_12 = ""

        try:
            res_check_13 = check_13(clear_table)
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=13))
            res_check_13 = "Не удалось"

        try:
            res_check_14, comment_14 = check_14(dict_text, clear_table)
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=14))
            res_check_14 = "Не удалось"
            comment_14 = ""

        try:
            res_check_15, res_check_16, comment_16 = check_15_and_16(
                dict_text, clear_table
            )
        except Exception as err:
            LOGGER.debug(log.format(err=err, num="15, 16"))
            res_check_15 = res_check_16 = comment_16 = "Не удалось"

        try:
            res_check_18, comment_18 = check_18(
                dict_text, clear_table, table_doc.columns
            )
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=18))
            res_check_18 = "Не удалось"
            comment_18 = ""

        try:
            res_check_20, comment_20 = check_20(table_doc, sps[61], dict_text, sps[62])
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=20))
            res_check_20 = "Не удалось"
            comment_20 = ""

        try:
            res_check_21 = check_21(dict_text)
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=21))
            res_check_21 = "Не удалось"

        try:
            res_check_22, res_check_24, comment_22, comment_24 = check_22(
                dict_text, clear_table
            )
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=22))
            res_check_22 = "Не удалось"
            res_check_24 = "Не удалось"
            comment_22 = ""
            comment_24 = ""
        try:
            res_check_23, comment_23 = check_23(table_doc)
        except Exception as err:
            LOGGER.debug(log.format(err=err, num=23))
            res_check_23 = "Не удалось"
            comment_23 = ""

    else:
        # res_check_12 = res_check_13 = res_check_14 = res_check_15 = res_check_16 = res_check_18 = res_check_22 =  "Не удалось прочитать таблицу"

        res_check_10 = res_check_11 = res_check_12 = res_check_13 = res_check_14 = (
            res_check_15
        ) = res_check_16 = None
        res_check_18 = res_check_20 = res_check_21 = res_check_22 = res_check_23 = (
            res_check_24
        ) = None
        comment_10 = comment_11 = comment_12 = comment_14 = comment_15 = comment_16 = ""
        comment_18 = comment_20 = comment_22 = comment_23 = comment_24 = ""

    try:
        res_check_17 = check_17(dict_text)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=17))
        res_check_17 = "Не удалось"

    try:
        res_check_19, comment_19 = check_19(table_doc, dict_text, sps[8])
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=19))
        res_check_19 = "Не удалось"
        comment_19 = ""

    try:
        res_check_25 = check_25(dict_text)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=25))
        res_check_25 = "Не удалось"

    try:
        res_check_26, commnet_26 = check_26(dict_text)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=26))
        res_check_26 = False
        commnet_26 = "Не удалось"

    try:
        res_check_27, comment_27 = check_27(dict_text, fpath)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=27))
        res_check_27 = False
        comment_27 = "Не удалось"

    try:
        res_check_28,  comment_28 = check_28(dict_text)
    except Exception as err:
        LOGGER.debug(log.format(err=err, num=28))
        res_check_28 = False
        comment_28 = "Не удалось"

    ans = {
        0: {
            "name": "Проверка №0: Шаблон таблицы",
            "status": res_check_0,
            "coment": comment_0,
        },
        1: {"name": names_checks[1], "status": res_check_1, "coment": comment_1},
        2: {"name": names_checks[2], "status": res_check_2, "coment": comment_2},
        3: {"name": names_checks[3], "status": res_check_3, "coment": ""},
        4: {"name": names_checks[4], "status": res_check_4, "coment": comment_4},
        5: {"name": names_checks[5], "status": res_check_5, "coment": ""},
        6: {"name": names_checks[6], "status": res_check_6, "coment": ""},
        7: {"name": names_checks[7], "status": res_check_7, "coment": ""},
        8: {"name": names_checks[8], "status": res_check_8, "coment": comment_8},
        9: {"name": names_checks[9], "status": res_check_9, "coment": ""},
        10: {"name": names_checks[10], "status": res_check_10, "coment": comment_10},
        11: {"name": names_checks[11], "status": res_check_11, "coment": comment_11},
        12: {"name": names_checks[12], "status": res_check_12, "coment": comment_12},
        13: {"name": names_checks[13], "status": res_check_13, "coment": ""},
        14: {"name": names_checks[14], "status": res_check_14, "coment": comment_14},
        15: {"name": names_checks[15], "status": res_check_15, "coment": ""},
        16: {"name": names_checks[16], "status": res_check_16, "coment": comment_16},
        17: {"name": names_checks[17], "status": res_check_17, "coment": ""},
        18: {"name": names_checks[18], "status": res_check_18, "coment": comment_18},
        19: {"name": names_checks[19], "status": res_check_19, "coment": comment_19},
        20: {"name": names_checks[20], "status": res_check_20, "coment": comment_20},
        21: {"name": names_checks[21], "status": res_check_21, "coment": ""},
        22: {"name": names_checks[22], "status": res_check_22, "coment": comment_22},
        23: {"name": names_checks[23], "status": res_check_23, "coment": comment_23},
        24: {"name": names_checks[24], "status": res_check_24, "coment": comment_24},
        25: {"name": names_checks[25], "status": res_check_25, "coment": ""},
        26: {"name": names_checks[26], "status": res_check_26, "coment": commnet_26},
        27: {"name": names_checks[27], "status": res_check_27, "coment": comment_27},
        28: {"name": names_checks[28], "status": res_check_28, "coment": comment_28},
    }
    for k in ans:
        ans[k]["status"] = str(ans[k]["status"])

    LOGGER.info(f"Время обработки для id = {id}, t = {(time.time() - stime):.2f} c")
    return ans


# global sp1,sp2, sp3, sp4, sp5, sp5prep, sps
# sp1, sp2, sp3, sp4, sp5, sp5prep, sps  = read_splist()
# sp_manager = SpManager(path2directory(),dbase)
Spdbmanager = SpDbManager(path2directory(), dbase)

app = FastAPI(
    docs_url = None, 
    redoc_url=None,
    root_path=config["root_path"], 
    openapi_url="/openapi.json"
)

current_file = Path(__file__)
current_file_dir = current_file.parent
project_root = current_file_dir.parent
project_root_absolute = project_root.resolve()
static_root_absolute = project_root_absolute 

app.mount('/static', StaticFiles(directory=os.path.join(project_root_absolute, 'static')), name='static')
@app.get("/documents", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # Set a timeout for the request
        response = await asyncio.wait_for(call_next(request), timeout=360)
        return response
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"message": "Request timeout"})


@app.post("/check_pdf")
async def check_pdf(file: UploadFile = File(...), query_id: int = 0, token: str= "secret"):
    """
    Проверка входящего документа (file pdf)
    """
    # if token != "y8f8TElxSOWCB1eZOAm0r0w66P29WT6m":
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Invalid token"
    #     )
    platform = sys.platform
    if platform == "win32":
        save_dir = Path(".\\src\\app\\save\\")
    else:
        save_dir = Path("./src/app/save/")
    if not save_dir.exists():
        save_dir.mkdir()
    # Cохранить файл на диск
    file_bytes = await file.read()
    fpath = save_dir / file.filename
    with open(fpath, "wb") as f:
        f.write(file_bytes)

    try:
        # Set a timeout for the long-running task
        # ans = await check_document
        # loop = asyncio.get_event_loop()
        ans = await check_document(fpath, ner_pipeline, query_id)

        # ans = await asyncio.wait_for(check_document, timeout=5)
        ans["query_id"] = int(query_id)
        return JSONResponse(content=ans)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")

    # try:

    # except asyncio.TimeoutError:
    # raise HTTPException(status_code=504, detail="Request timeout")


# >curl localhost:8000/update_guide/ -F "id=1" -F "file=@C:\Project\documents\src\app\data\work_docs\pdf\1.pdf"


@app.get("/guide")
async def get_guide(spname: str):
    """
    Получить справочник в xlsx
    Список справочников ["SP1","SP2","SP34","SP5","SP5LIST","SP7","SP8", "SP6LIST","SP61","SP62","SP63", "SP64","SP65","SP66","SP67", "SP68","SP69", "SP610", "SP611", "SP612"]
    """
    try:
        fcsv = Spdbmanager.save_csv(spname)
    except AssertionError as err:
        return HTTPException(
            status_code=404, detail=f"Ошибка сохранении в csv: {str(err)}"
        )
    df = pd.read_csv(str(fcsv), sep=";", encoding="cp1251")
    stream = io.StringIO()
    df.to_excel("temp.xlsx", index=False)
    response = FileResponse(
        path="temp.xlsx",
        headers={"Content-Disposition": f'attachment; filename="{fcsv.stem}.xlsx"'},
    )
    return response


@app.get("/template")
async def get_pattern(spname):
    """
    Получить шаблон структуры справочника в xlsx.
    Список справочников ["SP1","SP2","SP34","SP5","SP5LIST","SP7","SP8", "SP6LIST","SP61","SP62","SP63", "SP64","SP65","SP66","SP67", "SP68","SP69", "SP610", "SP611", "SP612"]
    """
    try:
        fcsv = Spdbmanager.save_pattern(spname)
    except AssertionError as err:
        return HTTPException(
            status_code=404, detail=f"Ошибка сохранении в csv: {str(err)}"
        )
    df = pd.read_csv(str(fcsv), sep=";", encoding="cp1251")
    stream = io.StringIO()
    df.to_excel("temp.xlsx", index=False)
    response = FileResponse(
        path="temp.xlsx",
        headers={"Content-Disposition": f'attachment; filename="{fcsv.stem}.xlsx"'},
    )
    return response


@app.post("/upload")
async def upload(spname, file: UploadFile = File(...)):
    """
    Залить новый справочник в систему из xlsx
    """
    file_bytes = await file.read()
    stream = io.BytesIO(file_bytes)
    try:
        frame = pd.read_excel(stream, engine="openpyxl")
    except:
        return HTTPException(
            400,
            detail="Неверный формат файла справочника должен быть 'xlsx' в кодировке 'cp1251'",
        )
    if Spdbmanager.replace(frame, spname) is True:
        ans = {"status": "True"}
    else:
        ans = {"status": "Неудалось объединить справочники"}
    # sp1, sp2, sp3, sp4, sp5, sp5_list_prep, sps = read_splist()
    return JSONResponse(content=ans)


@app.post("/append")
async def append(spname, file: UploadFile = File(...)):
    """
    Добавить новые записи из файла xlsx в справочник spname
    """
    file_bytes = await file.read()
    # data = json.loads(file_bytes)
    stream = io.BytesIO(file_bytes)
    try:
        frame = pd.read_excel(stream, engine="openpyxl")
    except:
        return HTTPException(
            400,
            detail="Неверный формат файла справочника должен быть 'xlsx' в кодировке 'cp1251'",
        )
    # make_backup(path2directory(), path2backup())
    if Spdbmanager.append(frame, spname) is True:
        ans = {"status": "True"}
    else:
        ans = {"status": "Неудалось объединить справочники"}
    # sp1, sp2, sp3, sp4, sp5, sp5_list_prep, sps = read_splist()
    return JSONResponse(content=ans)


@app.get("/read_backup")
async def read_backup():
    """
    Получить список папок c резервными копиями(backups). Название выбранной папки используется для восставновления в методе (restore)
    """
    list_backups = [f.name for f in Path(path2backup()).glob("*") if f.is_dir()]
    ans = {"status": True, "dirs": list_backups}
    return JSONResponse(content=list_backups)


@app.post("/restore")
async def restore(dir_name: str):
    """
    Восставновить справочники из выбранной папки backup. Параметр dir_name название папки для восстановления
    """
    backup_dir = os.path.join(path2backup(), dir_name)

    try:
        copy_pkl_files(backup_dir, path2directory())
        ans = {"status": True, "comment": "Восстановлен"}
    except:
        ans = {"status": False, "comment": "Неудалось восстановить справочники"}
    return JSONResponse(content=ans)


@app.get("/pdf_test")
async def pdf_test():
    fpath = os.getcwd() + "/src/app/data/test1.pdf"
    answer = {
        0: {"name": "Проверка №0: Шаблон таблицы", "status": True},
        1: {"name": names_checks[1], "status": False},
        2: {"name": names_checks[2], "status": False},
        3: {"name": names_checks[3], "status": False},
        4: {"name": names_checks[4], "status": True},
        5: {"name": names_checks[5], "status": True},
        6: {"name": names_checks[6], "status": True},
        7: {"name": names_checks[7], "status": True},
        8: {"name": names_checks[8], "status": True},
        9: {"name": names_checks[9], "status": True},
        10: {"name": names_checks[10], "status": True},
        11: {"name": names_checks[11], "status": "Не удалось"},
        12: {"name": names_checks[12], "status": False},
        13: {"name": names_checks[13], "status": True},
        14: {"name": names_checks[14], "status": True},
        15: {"name": names_checks[15], "status": True},
        16: {"name": names_checks[16], "status": True},
        17: {"name": names_checks[17], "status": False},
        18: {"name": names_checks[18], "status": True},
        19: {"name": names_checks[19], "status": "Не применимо"},
        20: {"name": names_checks[20], "status": "Не применимо"},
        21: {"name": names_checks[21], "status": True},
        22: {"name": names_checks[22], "status": True},
        23: {"name": names_checks[23], "status": True},
        24: {"name": names_checks[24], "status": True},
        25: {"name": names_checks[25], "status": True},
        26: {"name": names_checks[26], "status": True},
        27: {"name": names_checks[27], "status": True},
        28: {"name": names_checks[28], "status": True},
    }
    result = check_document(fpath, ner_pipeline, 0)
    ans = {}
    for k, v in answer.items():
        ans[k] = {"результат": result[k]["status"], "истинный ответ": v["status"]}
    return JSONResponse(content=ans)


@app.post("/convert/")
async def convert(file: UploadFile = File(...), id: int = -1):
    """
    Конвертировать справочник из file (pdf, xlsx) в {id}-номер справочника в excel.
    На выходе сконвертированный справочник xlsx для загрузки в систему.
    """
    id = int(id)
    platform = sys.platform

    if not allowed_file(file.filename):
        raise HTTPException(
            400,
            detail="Неверный формат файла справочника должен быть 'pdf','cvs','xls','xlsx'",
        )

    if platform == "win32":
        save_dir = Path(".\\src\\app\\sps\\convert")
    else:
        save_dir = Path("./src/app/sps/convert")
    if not save_dir.exists():
        save_dir.mkdir()
    files = Path(save_dir).glob("*")
    for f in files:
        os.remove(f)
    file_bytes = await file.read()
    fpath = save_dir / file.filename
    with open(fpath, "wb") as f:
        f.write(file_bytes)
    if id == 1:
        try:
            # raise Exception("Test №1")
            status, fps = sp_converter[1](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 2:
        try:
            status, fps = sp_converter[2](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 3:
        try:
            status, fps = sp_converter[3](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 5:
        try:
            status, fps = sp_converter[5](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 6:
        try:
            status, fps = sp_converter[6](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 7:
        try:
            status, fps = sp_converter[7](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 8:
        try:
            status, fps = sp_converter[8](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 9:
        try:
            status, fps = sp_converter[9](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 10:
        try:
            status, fps = sp_converter[10](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 11:
        try:
            status, fps = sp_converter[11](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 12:
        try:
            status, fps = sp_converter[12](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 13:
        try:
            status, fps = sp_converter[13](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 14:
        try:
            status, fps = sp_converter[14](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 15:
        try:
            status, fps = sp_converter[15](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 16:
        try:
            status, fps = sp_converter[16](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 17:
        try:
            status, fps = sp_converter[17](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 18:
        try:
            status, fps = sp_converter[18](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 19:
        try:
            status, fps = sp_converter[19](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 20:
        try:
            status, fps = sp_converter[20](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")
    elif id == 21:
        try:
            status, fps = sp_converter[21](save_dir, file.filename)
        except Exception as err:
            LOGGER.debug(f"Не удалось конвертировать справочник №{id}:{err}")

    save_dir = path2save()
    with zipfile.ZipFile(Path(save_dir) / "files.zip", mode="w") as zf:
        for f in fps:
            df = pd.read_pickle(f)
            excel_path = Path(save_dir) / f"{Path(f).stem}.xlsx"
            df.to_excel(excel_path, index=False)
            zf.write(excel_path)
    response = FileResponse(
        path=str(Path(save_dir) / "files.zip"),
        headers={"Content-Disposition": f'attachment; filename="files.zip"'},
    )
    return response


@app.get("/summary")
async def summary():
    """
    Краткая статистика по справочникам в системе
    """
    summary = Spdbmanager.summary()
    return JSONResponse(content=summary)


@app.post("/create_backup")
async def create_backup():
    """
    Создать бэкап db
    """
    new_dir = make_backup(path2directory(), path2backup())
    ans = {"status": True, "comment": f"Резерваная копия {new_dir} создана"}
    return JSONResponse(content=ans)


@app.post("/test/")
async def test():
    ans = {"ans": "Hello word"}
    return JSONResponse(content=ans)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9005)
