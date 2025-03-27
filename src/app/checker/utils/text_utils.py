import fitz
import regex
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import spacy
import pymorphy2

from sentence_transformers import SentenceTransformer, util
import pandas as pd

import re
from functools import reduce
import operator
import numpy as np
import pdfplumber
import dateparser
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    DatesExtractor,
    AddrExtractor,
    Doc,
)
import pickle
from pathlib import Path
import mapply
from concurrent.futures import ThreadPoolExecutor, as_completed

from fuzzywuzzy import fuzz

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)
MORPH = pymorphy2.MorphAnalyzer()
STOP_WORDS = stopwords.words('russian')


def preprocess_and_tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    
    words = text.split()

    processed_words = {
        MORPH.parse(word)[0].normal_form  
        for word in words
        if word not in STOP_WORDS and len(word) > 2  
    }
    
    return processed_words

def delete_data_fio(text: string, fio_ind: bool):
    """
    удалить все даты и последнее fio, пунктуацию
    Parameters
    ----------
    text

    Returns
    -------

    """
    text_dates = list(dates_extractor(text))
    for date in text_dates:
        text = text.replace(text[date.start : date.stop], "")
    if fio_ind:
        fio = list(names_extractor(text))[-1]
        text = text.replace(text[fio.start : fio.stop], "")
    text = remove_punctuation(text)
    return text


def get_correct_date(date):
    return ".".join([str(date.fact.day), str(date.fact.month), str(date.fact.year)])


def find_number_page(text: str, sub_text: str, index=0):
    number_pg_res = []
    for number_pg in range(len(text)):
        if re.search(sub_text, text[number_pg].lower()):
            # if sub_text in text[number_pg].lower():
            number_pg_res += [number_pg]

    if not number_pg_res:
        return

    # print(f'status: OK, pages: {number_pg_res}', end='\n')

    return number_pg_res[index]


def find_number_page_dict(text: dict, sub_text: str, index=0):
    # Ищем страницы, на которых встречается sub_text
    number_pg_res = []
    # Проходим по ключам (номерам страниц) и ищем в тексте на этих страницах
    for number_pg, page_text in text.items():
        if re.search(sub_text, page_text.lower()):
            number_pg_res.append(number_pg)
    if not number_pg_res:
        return None  # Возвращаем None, если ничего не найдено
    return number_pg_res[index]


def find_number_end_page(text: str, index=-1):
    END_PAGE_PATTERN = (
        "[в|В][-\w №()А-я]{1,100}опис[-\w\d\D №()А-я]{1,100}внесен[-\w\d\D №()А-я]"
    )
    number_pg_res = []
    for number_pg in range(len(text)):
        if re.search(END_PAGE_PATTERN, text[number_pg].lower()):
            number_pg_res += [number_pg]

    if not number_pg_res:
        return

    return number_pg_res[index]


def find_number_page_table(text: str, index=0):
    number_pg_res = []
    for number_pg in range(len(text)):
        if re.search("УТВЕРЖДАЮ", text[number_pg]):
            # if sub_text in text[number_pg].lower():
            number_pg_res += [number_pg]

    if not number_pg_res:
        return

    # print(f'status: OK, pages: {number_pg_res}', end='\n')

    return number_pg_res[index]


def read_text_documents(path: str):
    """
    Прочитать текст документа
    Returns
    ----------
    text: dict[int:string], словарь по страницам документа
    full_text: string, полный текст документа
    """
    text = {}
    full_text = ""
    page_blocks = []
    with fitz.open(path) as doc:
        for num, page in enumerate(doc.pages()):
            text[num] = page.get_text()
            page_blocks += [page.get_text("dict")["blocks"]]
            full_text += " " + page.get_text()
    return text, full_text, page_blocks


def clear_text(text: str, simbol: str):
    """
    Очистить текст от символа "simbol"
    Returns
    ----------
    text: string, текст очищен от simbol
    """

    def check_null(x):
        return x != ""

    pr_text = text.replace(simbol, " ")

    filtered_names = filter(check_null, pr_text.split(" "))
    pr_text = " ".join(list(filtered_names))

    return pr_text


def clear_text_abc(text):
    l_text = text.split(" ")[::-1]

    ind = 0
    for word in l_text:
        if len(word) > 1:
            break
        ind += 1

    l_text = l_text[ind:]
    return " ".join(l_text[::-1])


def get_number_fond(text: dict, number_page: int):
    """
    Получить номер фонда из документа
    Returns
    ----------
    nums: int, номер фонда
    """
    pr_text = text[number_page]
    for symbol in ["\n", " "]:
        pr_text = clear_text(pr_text, symbol)

    name_fonds = ["фонд", "оаф"]
    number_fond = ""

    for fond in name_fonds:
        if pr_text.lower().find(fond) + 1:
            number_fond = pr_text[
                pr_text.lower().find(fond) : pr_text.lower().find(fond) + 100
            ]

    nums = re.findall(r"\d+", number_fond)
    nums = [i for i in nums]
    if not nums:
        return None
    return nums[0]


def get_info_opice(text: dict, number_page: int):
    """
    Получить номер описи из документа
    Returns
    ----------
    nums: int, номер фонда
    """
    pr_text = text[number_page]

    info_opice = pr_text[
        pr_text.lower().find("опись") : pr_text.lower().find("опись") + 100
    ].split("\n")

    for symbol in ["\n", " "]:
        info_opice[0] = clear_text(info_opice[0], symbol)
        info_opice[1] = clear_text(info_opice[1], symbol)

    nums = re.findall(r"\d+", info_opice[0])
    nums = [i for i in nums]

    number_opice = nums[0]
    type_opice = info_opice[1]

    return number_opice, type_opice


def get_similar_3(text_1: str, text_2: str):
    return fuzz.ratio(text_1, text_2)


def remove_punctuation(dirty_text):
    """
    Убрать пунктуацию из текста
    Returns
    ----------
    clean_text: string, текст без знаков пунктуации
    """
    clean_text = "".join(
        [ch for ch in dirty_text if ch not in string.punctuation]
    )  # метод для удаления пунктуации
    clean_text = re.sub(r"[^\w\s]+|[\d]+", "", clean_text)
    clean_text = clear_text(clean_text, " ")
    return clean_text


def get_name_fond(text, page_blocks):
    """
    Получить имя фонда из текста первой страницы
    Returns
    ----------
    name_fond: string, имя фонда
    """

    pr_text = text[0]
    pr_text = pr_text[: pr_text.lower().find("опись")]

    list_filter_name = ["", "ОАФ", "ОПИСЬ", "ФОНД"]

    def filter_space(x):
        return (x not in list_filter_name) and (len(x) != 1)

    bold_text = ""
    for blocks in page_blocks[0]:
        res_block = blocks["lines"][0]["spans"][0]
        if "bold" in res_block["font"].lower():
            bold_text += " " + res_block["text"]

    bold_text = bold_text.strip()

    list_texts = [
        " ".join(regex.findall(r"\b[[:upper:]]+\b", text.strip()))
        for text in pr_text.split("\n \n")
    ]

    list_texts_result = list(filter(filter_space, list_texts))

    name_fond_res = "ERROR: Имя фонда не найдено"
    if list_texts_result:
        name_fond = (" ".join(list_texts_result)).replace("ОАФ", "").replace("ФОНД", "")
        name_fond_res = " ".join(
            [word.strip() for word in name_fond.split(" ") if word.strip() != ""]
        )

    if bold_text:
        return bold_text

    return clear_text_abc(name_fond_res)


def get_name_fond_table(text, number):
    """
    Получить имя фонда из текста выбранной страницы (для таблицы)
    Returns
    ----------
    name_fond: string, имя фонда
    """
    pr_text = text[number]

    left_line = pr_text.lower().find("опись")

    pr_text = pr_text[:left_line]

    def filter_space(x):
        return x not in [""]

    list_texts = [
        " ".join(regex.findall(r"\b[[:upper:]]+\b", text.strip()))
        for text in pr_text.split("\n \n")
    ]

    list_texts_result = list(filter(filter_space, list_texts))

    name_fond_res = "ERROR: Имя фонда не найдено"
    if list_texts_result:
        name_fond = (
            (" ".join(list_texts_result))
            .replace("УТВЕРЖДАЮ", "")
            .replace("ОАФ", "")
            .replace("ФОНД", "")
        )
        name_fond_res = " ".join(
            [word.strip() for word in name_fond.split(" ") if word.strip() != ""]
        )

    return clear_text_abc(name_fond_res)


def get_key_words(name_fond, sp3):
    """
    Получить ключевые слова из имени фонда
    Returns
    ----------
    key_words: list, списко ключевых слов
    """
    sp3["Наименование"] = sp3["Наименование"].str.lower().str.strip()

    tokens = word_tokenize(name_fond)

    stemmer = SnowballStemmer("russian")
    tokens_stem = [stemmer.stem(token) for token in tokens]

    stop_words = stopwords.words("russian")
    sen = " ".join([token for token in tokens_stem if token not in stop_words])
    sen = remove_punctuation(sen)

    list_words = list(set(sen.split(" ")))

    words_cnt = []
    for word in list_words:
        words_cnt += [len(sp3[sp3["Наименование"].str.contains(word)])]

    df = pd.DataFrame()
    df["words"] = list_words
    df["cnt"] = words_cnt
    df = df[df["cnt"] != 0].sort_values(by="cnt")
    if df.empty:
        return []
    key_words = df["words"].to_list()
    count = 3 if len(key_words) > 3 else len(key_words)
    return key_words[:count]


def get_similar(text1, ner_pipeline):
    results = ner_pipeline(text1.strip("."))
    index = -1
    name = ""
    org = []

    flags = []
    for entity in results:
        if entity["entity"] == "B-LOC" or entity["entity"] == "I-LOC":
            flags.append(entity["word"])
    if flags:
        flag_start = flags[0][1:]
        flag_end = flags[-1][1:]
        index_s = text1.find(flag_start)
        index_e = text1.find(flag_end) + len(flag_end)

        if index_s != -1 and index_e != -1:
            name = text1[index_s : index_e + 1]

    # Находим организацию
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text1)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            org.append(ent.text)
    org = " ".join(org)
    return name, org


def check_sim(text1, text2):
    if text1.startswith(" "):
        text1 = text1[1:]
    if text2.startswith(" "):
        text2 = text2[1:]
    # удаляем инициалы
    text1 = re.sub(r"\b\w{2}\b", "", text1)
    text2 = re.sub(r"\b\w{2}\b", "", text2)
    text1 = re.sub(r"\b\w{1}\b", "", text1)
    text2 = re.sub(r"\b\w{1}\b", "", text2)

    name1 = text1.strip(".").lower()
    name2 = text2.strip(".").lower()
    # удаляем первое слово
    name1 = re.sub(r"^\S+\s", "", name1)

    # удаляем профессии
    list_delete = ['редактор']
    for elem in list_delete:
        name1 = name1.replace(elem, "")

    morph = pymorphy2.MorphAnalyzer()
    org_lemmas = {
        morph.parse(word)[0].normal_form for word in nltk.word_tokenize(name1)
    }
    text2_lemmas = {
        morph.parse(word)[0].normal_form for word in nltk.word_tokenize(name2)
    }

    if org_lemmas != set() and org_lemmas == text2_lemmas:
        return True, org_lemmas, text2_lemmas
    else:
        return False, org_lemmas, text2_lemmas


def delete_data(text: string, fio_ind: bool):
    """
    удалить все даты
    """
    text_dates = list(dates_extractor(text))
    for date in text_dates:
        text = text.replace(text[date.start : date.stop], "")
    text = remove_punctuation(text)
    return text


def get_similar_2(model, embedding_1, embedding_2):
    """
    Получить степень схожести текстов
    подготовка данных вынесена отдельно для снижения сложности алгоритма
    Returns
    ----------
    cos_sim:
    """

    # embedding_1 = model.encode(text1.lower(), convert_to_tensor=True)
    # embedding_2 = model.encode(text2.lower(), convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2)


def check_names_fonds(name_fond, sp3):
    sp3_mod = sp3.copy()
    sp3_mod["Наименование"] = sp3_mod["Наименование"].str.lower().str.strip()

    key_words = get_key_words(name_fond, sp3_mod)

    if len(list(key_words)) < 1:
        return False, []

    check_sp3 = [sp3_mod["Наименование"].str.contains(x) for x in key_words]

    check_res = check_sp3[0]
    names_fonds = []
    for check in check_sp3[1:]:
        check_res = check_res & check
        names_fonds += sp3_mod[check_res]["Наименование"].to_list()

    names_fonds = list(map(lambda x: clear_text(x, "\n"), names_fonds))

    if len(names_fonds) <= 1:
        return False, names_fonds

    prob_similar = []
    for name in names_fonds:
        prob_similar += [get_similar(name, name_fond).item()]

    names = []
    if not len(prob_similar):
        return False, []
    if len(prob_similar) > 5:
        for i in sorted(prob_similar)[-5:]:
            names += [names_fonds[prob_similar.index(i)]]
        if sorted(prob_similar)[-1] > 0.7:
            return False, names[::-1]
        return False, []

    for i in sorted(prob_similar):
        names += [names_fonds[prob_similar.index(i)]]
    if sorted(prob_similar)[-1] > 0.7:
        return False, names[::-1]
    return False, []


def get_begin_table(tables):
    y_begin = 0
    y_end = 0
    used_tables = []
    if len(tables) > 2:
        try:
            for table in tables:
                for line in table.extract():
                    filtered_list = [item for item in line if item is not None]
                    line_lower = "".join(filtered_list).lower()
                    if '№' in line_lower and "загол" in line_lower:
                        used_tables.append(table)
        except:
            pass
    else:
        used_tables = tables
    
    for table in used_tables:
        y_b = np.array(table.bbox[1]).mean()
        y_e = np.array(table.bbox[3]).mean()
        if y_b > y_begin:
            y_begin = y_b
            y_end = y_e
        return table, y_begin, y_end
    

def to_date(string):
    """
    Выделить диапазон дат из строки "string" таблицы
    """
    if len(string) > 6:
        string = re.sub("\s+", "", string)
    # string = string.replace(" ","")
    string = string.strip()
    string = string.replace("–", "-")
    string = string.replace("--", "-")
    left_date = None
    right_date = None

    months = [
        "январь",
        "февраль",
        "март",
        "апрель",
        "май",
        "июнь",
        "июль",
        "август",
        "сентябрь",
        "октябрь",
        "ноябрь",
        "декабрь",
    ]
    has_russian_month = lambda s: any([True if m in s else False for m in months])
    if has_russian_month(string):
        f_year = int(re.findall(r"\d+", string)[0])
        if "-" in string:
            dates = re.findall(r"\w+", string.replace(" ", ""))
            if len(dates) == 2:
                left_date = pd.to_datetime(
                    dateparser.parse(
                        dates[0],
                        settings={
                            "RELATIVE_BASE": datetime(year=f_year, month=1, day=1)
                        },
                    )
                )
                right_date = pd.to_datetime(
                    dateparser.parse(
                        dates[1],
                        settings={
                            "RELATIVE_BASE": datetime(year=f_year, month=1, day=1)
                        },
                    )
                )
                return left_date.to_pydatetime(), right_date.to_pydatetime()
        else:
            m = np.argmax([1 if m in string else 0 for m in months])
            dates = re.findall(r"\w+", string.replace(" ", ""))
            left_date = pd.to_datetime(
                dateparser.parse(
                    dates[0],
                    settings={"RELATIVE_BASE": datetime(year=f_year, month=m, day=1)},
                )
            )
            right_date = pd.to_datetime(
                dateparser.parse(
                    dates[0],
                    settings={"RELATIVE_BASE": datetime(year=f_year, month=m, day=2)},
                )
            )
            # print('r:=',string, left_date, right_date)
            return left_date.to_pydatetime(), right_date.to_pydatetime()
    if len(string) > 5:
        if "-" in string:
            a, b = string.split("-")
            # print(a,b)
            left_date = pd.to_datetime(a, dayfirst=True)
            right_date = pd.to_datetime(b, dayfirst=True)
            # print(string, left_date, right_date)

        elif "/" in string:
            dates = [pd.to_datetime(t).to_pydatetime() for t in string.split("/")]
            return dates[0], dates[-1]
        elif " " in string:
            a, b = string.split(" ")
            left_date = pd.to_datetime(a, dayfirst=True)
            right_date = pd.to_datetime(b, dayfirst=True)
        else:
            return (
                pd.to_datetime(string, dayfirst=True).to_pydatetime(),
                pd.to_datetime(string, dayfirst=True).to_pydatetime(),
            )
    else:
        return (
            pd.to_datetime(string, dayfirst=True).to_pydatetime(),
            pd.to_datetime(string, dayfirst=True).to_pydatetime(),
        )
    return left_date.to_pydatetime(), right_date.to_pydatetime()


def get_date_inventory(dict_text):
    """
    Получить диапазон даты описи из документа
    """
    p = find_number_page(dict_text, "утверждаю\s")
    txt = dict_text[p]
    ans = re.findall("опись[\s\S\d]{1,150}[\d]{4}[\s\S]{0,5}год", txt.lower())
    if len(ans) > 0:
        sub_str = ans[0]
    else:
        s = txt.lower().find("опись")
        if s < 0:
            s = txt.lower().find("о п и с ь")
        end = txt.find("п/п")
        end2 = txt.lower().find("индекс")
        if end2 > end and end > 0:
            end = int((end + end2) / 2)
        elif end == -1 and end2 > 0:
            end = end2
        sub_str = txt[s:end]

    if "за" in sub_str and "год" in sub_str:
        ss = sub_str.lower().find("за ")
        ee = len(sub_str)
        for x in re.finditer("\d{4}", sub_str):
            ee = x.span()[1]

        r = re.findall(r"\d{4}-\d{4}", sub_str)
        if len(r) > 0:
            doc_ldate, doc_rdate = to_date(r[0])
            # doc_rdate += relativedelta(years=1)
            return doc_ldate, doc_rdate

        if "утверждаю" in sub_str[ss + 2 : ee].lower():
            ee = sub_str.lower().find("утверждаю")
        doc_ldate, doc_rdate = to_date(sub_str[ss + 2 : ee])
        if "/" in sub_str[ss + 2 : ee]:
            doc_rdate += relativedelta(years=1)
            return doc_ldate, doc_rdate
    else:
        ee = sub_str.lower().find("n\n")
        doc_ldate, doc_rdate = to_date(sub_str[s + 4 : ee])
    if doc_ldate == doc_rdate:
        doc_rdate = doc_ldate
    return doc_ldate, doc_rdate


def add_one_month(orig_date):
    # advance year and month by one month
    new_year = orig_date.year
    new_month = orig_date.month + 1
    # note: in datetime.date, months go from 1 to 12
    if new_month > 12:
        new_year += 1
        new_month -= 12

    last_day_of_month = calendar.monthrange(new_year, new_month)[1]
    new_day = min(orig_date.day, last_day_of_month)

    return orig_date.replace(year=new_year, month=new_month, day=new_day)


def extract_proper_dates(text):
    if "ОПИСЬ" in text:
        text = text.split("ОПИСЬ", 1)[1]
    # Регулярное выражение для поиска даты в формате "дата / месяц/ год"
    date_pattern1 = r"\b\d{1,2}\s(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s\d{4}\b"
    lines = text.splitlines()
    # Проходим по строкам и ищем первую строку с датой
    flag = None
    for line in lines:
        if re.search(date_pattern1, line):
            date_str = line  # первая строка с датами
            date_pattern2 = r"(\d{1,2}\s(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s\d{4})(\*)?"
            break
    else:
        flag = 0
    if flag == 0:
        date_pattern1 = r"\b\d{4}\b"
        lines = text.splitlines()
        for line in lines:
            if re.search(date_pattern1, line):
                date_str = line  # первая строка с датами
                date_pattern2 = r"\b(\d{4})(\*)?"
                break

    # Регулярное выражение для поиска дат на титуле в формате "дата / месяц/ год" с возможным наличием `*`
    # date_pattern2 = r"(\d{1,2}\s(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s\d{4})(\*)?"
    matches2 = re.findall(date_pattern2, text)
    date1 = int(matches2[0][0][-4:])
    mark1 = matches2[0][1]
    if len(matches2) != 1:
        date2 = int(matches2[1][0][-4:])
        mark2 = matches2[1][1]
    elif len(matches2) == 1 and matches2[0][1] == "":
        date2 = date1
        mark2 = ""
    else:
        date2 = datetime.now().year
        mark2 = ""

    # список возможных дат в предисловии
    if mark1 == "" and mark2 == "":
        proper_dates = list(range(date1, date2 + 1))
    elif mark1 != "" and mark2 == "":
        proper_dates = sorted(list(range(date1 - 1, 1950 - 1, -1))) + list(
            range(date1, date2 + 1)
        )
    elif mark1 == "" and mark2 != "":
        proper_dates = list(range(date1, date2)) + list(
            range(date2, datetime.now().year + 1)
        )
    else:
        proper_dates = (
            list(range(date1 - 1, 1950 - 1, -1)).sort()
            + list(range(date1 + 1, date2 + 1))
            + list(range(date2, datetime.now().year + 1))
        )
    return proper_dates


def lemmatize_text(text, morph):
    """
    Функция для лемматизации текста

    Returns
    ---------
    Лематизированная строка
    """
    words = text.split()  # Разделяем текст на слова
    lemmatized_words = [
        morph.parse(word)[0].normal_form for word in words
    ]  # Лемматизация
    return " ".join(lemmatized_words)  # Объединяем обратно в строку


def remove_stopwords(text):
    """
    Функция для удаления стоп-слов

    Returns
    ---------
    Строка
    """
    stop_words = set(stopwords.words("russian"))
    words = word_tokenize(text)  # Токенизация текста
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def remove_numbers_and_symbols(text):
    return re.sub(r"[0-9№]", "", text)


def remove_words_after_year(sentence):
    """
    Функция для удаления слов после слова год

    """
    index = sentence.find("год")
    if index != -1:
        return sentence[: index + len("год")]
    return sentence


def find_nouns(words):
    """
    Функция для нахождения улючевых слов в тексте

    Returns:
    ----------
    result - список ключевых слов
    """
    first_noun = None
    second_noun = None
    first_noun_index = None

    nlp = spacy.load("ru_core_news_sm")
    # Регулярное выражение для поиска слова "отчёт" с любыми окончаниями
    report_pattern = re.compile(r"отчёт.*")  # Захватывает "отчёт" и любые его формы

    # Функция для лемматизации текста
    def lemmatize_text(word):
        doc = nlp(word)
        return [token.lemma_ for token in doc]

    # Поиск первого существительного
    for i, word in enumerate(words):
        lemmatized_word = lemmatize_text(word)[0]
        doc = nlp(lemmatized_word)

        for token in doc:
            # Проверяем наличие "отчёт"
            if report_pattern.fullmatch(token.text.lower()):
                # Проверяем предшествующее слово на наличие прилагательного
                if i > 0 and nlp(words[i - 1])[0].pos_ == "ADJ":
                    first_noun = words[i - 1] + " " + token.text
                else:
                    first_noun = token.text
                first_noun_index = i  # Сохраняем индекс первого существительного
                break

            if token.pos_ == "NOUN":
                if i > 0 and nlp(words[i - 1])[0].pos_ == "ADJ":
                    first_noun = words[i - 1] + " " + token.text
                else:
                    first_noun = token.text
                first_noun_index = i  # Сохраняем индекс первого существительного
                break
        if first_noun:
            break

    # Проверяем наличие союза "и" и находим второе существительное, если расстояние меньше или равно 5
    if "и" in words and first_noun_index is not None:
        index_of_and = words.index("и")
        if index_of_and - first_noun_index > 5:
            return [first_noun]  # Возвращаем только первое существительное

        for j in range(index_of_and + 1, len(words)):
            next_word = words[j]
            lemmatized_next_word = lemmatize_text(next_word)[0]
            next_doc = nlp(lemmatized_next_word)

            for next_token in next_doc:
                # Проверяем наличие "отчёт" после "и"
                if report_pattern.fullmatch(next_token.text.lower()):
                    # Проверяем предшествующее слово на наличие прилагательного
                    if j > index_of_and + 1 and nlp(words[j - 1])[0].pos_ == "ADJ":
                        second_noun = words[j - 1] + " " + next_token.text
                    else:
                        second_noun = next_token.text
                    break

                elif next_token.pos_ == "NOUN":
                    if j > index_of_and + 1 and nlp(words[j - 1])[0].pos_ == "ADJ":
                        second_noun = words[j - 1] + " " + next_token.text
                    else:
                        second_noun = next_token.text
                    break
            if second_noun:
                break

    # Формируем список результатов
    result = []
    if first_noun:
        result.append(first_noun)
    if second_noun:
        result.append(second_noun)

    return result


def prep_dataset_doc(table):
    table.columns.values[0] = "№ п/п"
    table.columns.values[2] = "Заголовок дела"
    table.columns.values[3] = "Крайние даты"
    table.columns = table.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    indices_to_drop = table[table["№ п/п"] == ""].index
    table = table.drop(indices_to_drop)
    table = table[table["Заголовок дела"].str.len() > 2]
    morph = pymorphy2.MorphAnalyzer()
    table = table.drop(table[table["Крайние даты"] == ""].index).reset_index()
    table["Заголовок дела"] = table["Заголовок дела"].str.replace(
        "П остановления", "Постановления"
    )
    table["Заголовок дела"] = table["Заголовок дела"].str.replace(
        f"[{string.punctuation}]", "", regex=True
    )
    table["Заголовок дела"] = table["Заголовок дела"].apply(
        lambda x: lemmatize_text(
            remove_stopwords(
                remove_words_after_year(remove_numbers_and_symbols(x.lower()))
            ),
            morph,
        )
    )
    table["Флаг"] = None
    return table


def prep_dataset_sp5(doc_table):
    """
    Предобработка СП5 для 10 проверки
    """
    morph = pymorphy2.MorphAnalyzer()
    doc_table["Вид документа"] = doc_table["Вид документа"].apply(
        lambda x: lemmatize_text(remove_stopwords(x.lower()), morph)
    )
    doc_table = doc_table.replace("\n", " ", regex=True)
    doc_table["Вид документа"] = doc_table["Вид документа"].str.replace(
        f"[{string.punctuation}]", "", regex=True
    )
    return doc_table


def prep_table(table):
    """
    Обработка данных таблицы описи
    """

    def extract_until_second_capitalized(title):
        words = title.split()
        result = []
        capitalized_count = 0

        for word in words:
            if word[0].isupper():
                capitalized_count += 1
                if capitalized_count == 2:
                    break
            result.append(word)
        return " ".join(result)

    def truncate_sentence(sentence):
        pattern = r"\bкомисси\w*\b"
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return sentence[: match.end()]
        else:
            return sentence

    morph_analyzer = pymorphy2.MorphAnalyzer()

    table.columns.values[0] = "№ п/п"
    table.columns.values[2] = "Заголовок дела"
    table.columns.values[3] = "Крайние даты"
    table.columns = table.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    # table['Заголовок дела'] = table['Заголовок дела'].apply(extract_until_second_capitalized)
    # table['Заголовок дела'] = table['Заголовок дела'].apply(lambda x: truncate_sentence(x))
    indices_to_drop = table[table["№ п/п"] == ""].index
    table = table.drop(indices_to_drop)
    table = table[table["Заголовок дела"].str.len() > 2]
    table = table.drop(table[table["Крайние даты"] == ""].index).reset_index()
    table["Заголовок дела"] = table["Заголовок дела"].str.replace(
        "П остановления", "Постановления"
    )
    table["Заголовок дела"] = table["Заголовок дела"].str.replace(
        f"[{string.punctuation}]", "", regex=True
    )
    table["Заголовок дела"] = table["Заголовок дела"].apply(
        lambda x: lemmatize_text(
            remove_stopwords(
                remove_words_after_year(remove_numbers_and_symbols(x.lower()))
            ),
            morph_analyzer,
        )
    )
    table["Заголовок дела"] = table["Заголовок дела"].replace(
        "участый ков", "участковый", regex=True
    )
    table["Флаг"] = None
    return table


def remove_words_from_string(or_string, words_to_remove):
    """
    удаление слов из предложения
    """
    words = or_string.split()
    filtered_words = [
        word for word in words if word.lower() not in map(str.lower, words_to_remove)
    ]
    return " ".join(filtered_words)


def prep_sp(sp):
    morph = pymorphy2.MorphAnalyzer()
    sp.iloc[:, 0] = sp.iloc[:, 0].apply(
        lambda x: lemmatize_text(
            remove_stopwords(remove_numbers_and_symbols(x.lower())), morph
        )
    )
    sp = sp.replace("\n", " ", regex=True)
    sp.iloc[:, 0] = sp.iloc[:, 0].str.replace(f"[{string.punctuation}]", "", regex=True)
    words_to_remove = [
        "российский",
        "федерация",
        "челябинский",
        "область",
        "регион",
        "район",
        "президент",
        "рф",
    ]
    sp.iloc[:, 0] = sp.iloc[:, 0].apply(
        lambda x: remove_words_from_string(x, words_to_remove)
    )
    return sp


def find_flexible_word(text, phrase):
    words = phrase.split()
    if len(words) == 1:
        pattern = rf"\b{words[:-2]}[а-яА-Я]*\b"
        match_ = re.search(pattern, text, re.IGNORECASE)
    else:
        pattern = rf"\b{words[0][:-2]}[а-яА-Я]*\b\s+\b{words[1][:-2]}[а-яА-Я]*\b"
        match_ = re.search(pattern, text, re.IGNORECASE)
    return True if match_ else False


def prep_sp6_tables(dbase):
    """
    Для выполнения этой проверки в директории должны быть все таблицы из сп6, sp6_1, sp6_2 ... sp6_12
    """

    processed_dataframes = []
    for n in [
        "SP61",
        "SP62",
        "SP63",
        "SP64",
        "SP65",
        "SP66",
        "SP67",
        "SP68",
        "SP69",
        "SP610",
        "SP611",
        "SP612",
    ]:
        df = dbase.get_table(n)
        df.rename(columns={df.columns[0]: "Номер"}, inplace=True)
        df["Номер"] = df["Номер"].astype(str).str.rstrip(".")
        # Применяем вашу логику, проверяя наличие части слова "Пост"
        df["contains_flag"] = df.iloc[:, 2:].apply(
            lambda x: any("Пост" in str(val) for val in x.values), axis=1
        )
        processed_df = df[["Номер", df.columns[-1]]].copy()
        processed_df.reset_index(drop=True, inplace=True)
        processed_dataframes.append(processed_df)
    combined_df = pd.concat(processed_dataframes, ignore_index=True)

    return combined_df


def prep_sp_lists(frame):
    """
    Предобработка файлов sp5_list.pkl и sp6_list.pkl
    path2frame: путь к файлу справочника из src/app/sps
    f_save: "sp5_list_prep.pkl"
    """
    doc_table = frame.copy()
    morph = pymorphy2.MorphAnalyzer()
    doc_table.iloc[:, 0] = doc_table.iloc[:, 0].apply(
        lambda x: lemmatize_text(remove_stopwords(str(x).lower()), morph)
    )
    doc_table.iloc[:, 0] = doc_table.iloc[:, 0].str.replace(
        f"[{string.punctuation}]", "", regex=True
    )
    doc_table.iloc[:, 2] = doc_table.iloc[:, 2].apply(
        lambda x: lemmatize_text(str(x).lower(), morph)
    )
    # with open(Path(path2frame).parent / f_save ,'wb') as file:
    #     pickle.dump(doc_table, file)
    return doc_table


def prep_sp_lists_fast(frame, word_tokenize):
    """
    Предобработка для файлов sp5_list.pkl и sp6_list.pkl
    path2frame: путь к файлу справочника из src/app/sps
    name_save: "sp5_list_prep.pkl"
    """

    def lemmatize_text_remove(text, morph_analyzer, stop_words, word_tokenize):
        words = word_tokenize(text)  # Токенизация текста
        filtered_words = [word for word in words if word.lower() not in stop_words]
        text = " ".join(filtered_words)
        words = text.split()  # Разделяем текст на слова
        lemmatized_words = [
            morph_analyzer.parse(word)[0].normal_form for word in words
        ]  # Лемматизация
        return " ".join(lemmatized_words)  # Объединяем обратно в строку

    def lemmatize_text(text, morph_analyzer):
        words = text.split()  # Разделяем текст на слова
        lemmatized_words = [
            morph_analyzer.parse(word)[0].normal_form for word in words
        ]  # Лемматизация
        return " ".join(lemmatized_words)  # Объединяем обратно в строку

    doc_table = frame.copy()

    mapply.init(
        n_workers=4,
        chunk_size=200,
        max_chunks_per_worker=8,
        progressbar=False,
    )

    morph_analyzer = pymorphy2.MorphAnalyzer()
    stop_words = set(stopwords.words("russian"))
    # from nltk.tokenize import word_tokenize
    kwargs = {
        "morph_analyzer": morph_analyzer,
        "stop_words": stop_words,
        "word_tokenize": word_tokenize,
    }
    doc_table.iloc[:, 0] = doc_table.iloc[:, 0].mapply(
        lambda x: lemmatize_text_remove(str(x).lower(), **kwargs)
    )
    doc_table.iloc[:, 0] = doc_table.iloc[:, 0].str.replace(
        f"[{string.punctuation}]", "", regex=True
    )
    doc_table.iloc[:, 2] = doc_table.iloc[:, 2].apply(
        lambda x: lemmatize_text(str(x).lower(), morph_analyzer)
    )
    return doc_table


def prep_dataset_doc_20(table):
    table.columns.values[0] = "Номер"
    table.columns = table.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    indices_to_drop = table[table["Номер"] == ""].index
    table = table.drop(indices_to_drop)
    table = table[table["Заголовок единицы хранения"].str.len() > 2]
    table = table.reset_index()
    morph = pymorphy2.MorphAnalyzer()
    table["Заголовок единицы хранения"] = table[
        "Заголовок единицы хранения"
    ].str.replace("П остановления", "Постановления")
    table["Заголовок единицы хранения"] = table[
        "Заголовок единицы хранения"
    ].str.replace(f"[{string.punctuation}]", "", regex=True)
    table["Заголовок единицы хранения"] = table["Заголовок единицы хранения"].apply(
        lambda x: lemmatize_text(
            remove_stopwords(
                remove_words_after_year(remove_numbers_and_symbols(x.lower()))
            ),
            morph,
        )
    )
    table["Флаг"] = None
    return table


def prep_dataset_doc_10(table):
    table.rename(columns={table.columns[0]: "Номер"}, inplace=True)
    table.rename(columns={table.columns[2]: "Заголовок дела"}, inplace=True)
    table.rename(columns={table.columns[3]: "Крайние даты"}, inplace=True)
    table.columns = table.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    indices_to_drop = table[table["Номер"] == ""].index
    table = table.drop(indices_to_drop)
    indices_to_drop1 = table[table["Заголовок дела"].str.len() < 12].index
    table = table.drop(indices_to_drop1)
    indices_to_drop2 = table[table["Номер"].str.len() > 5].index
    table = table.drop(indices_to_drop2)
    morph = pymorphy2.MorphAnalyzer()
    table = table.drop(table[table["Крайние даты"] == ""].index).reset_index()
    table["Заголовок дела"] = table["Заголовок дела"].str.replace(
        "П остановления", "Постановления"
    )
    table["Заголовок дела"] = table["Заголовок дела"].str.replace(
        f"[{string.punctuation}]", " ", regex=True
    )
    # table['Заголовок дела'] = table['Заголовок дела'].apply(lambda x: lemmatize_text(remove_stopwords(x.lower())))
    table["Заголовок дела"] = table["Заголовок дела"].apply(
        lambda x: lemmatize_text(remove_stopwords(x.lower()), morph)
    )
    table["Флаг"] = None
    return table


def prep_sp5_table(df, flag):

    df.rename(columns={df.columns[0]: "Номер"}, inplace=True)
    df["Номер"] = df["Номер"].str.rstrip(".")
    if flag == 1:
        df["contains_flag"] = df.iloc[:, 2:].apply(
            lambda x: any("Пост" in str(val) for val in x.values), axis=1
        )

    else:
        words_to_check = ["50 лет", "70 лет", "50/75 лет"]
        df["contains_flag"] = df.iloc[:, 2:].apply(
            lambda x: any(
                any(word in str(val) for word in words_to_check) for val in x.values
            ),
            axis=1,
        )

    processed_df = df[["Номер", df.columns[-1]]].copy()
    processed_df.reset_index(drop=True, inplace=True)
    return processed_df


def process_row_10(i, table, doc_table, comments):
    """
    Функция для обработки строки и поиска соответствий.
    """
    words1 = table["Заголовок дела"][i].lower().split()
    key_words = words1
    first_two = words1[:2]
    max_intersection_count = 0
    if set(first_two) == {"проектный", "задание"}:
        table.loc[i, "Флаг"] = 1

    if set(first_two) == {"бухгалтерский", "отчёт"}:
        table.loc[i, "Флаг"] = 1

    if "проект" in key_words:
        first_two[0] == "проект"

    if table["Флаг"][i] != 1:
        for j in range(len(doc_table["Название"])):
            # Проверки на совпадение с документами
            if doc_table["Раздел"][j][0] in set(words1):
                words2 = set(doc_table["Название"][j])
                intersection_count = len(set(key_words).intersection(words2))
                if (
                    intersection_count != 0
                    and max_intersection_count < intersection_count
                ):
                    max_intersection_count = intersection_count
                    table.loc[i, "Флаг"] = 1
                    break

            if len(doc_table["Раздел"][j]) == 2 and doc_table["Раздел"][j][1] in set(
                words1
            ):
                words2 = set(doc_table["Название"][j])
                intersection_count = len(set(key_words).intersection(words2))
                if (
                    intersection_count != 0
                    and max_intersection_count < intersection_count
                ):
                    max_intersection_count = intersection_count
                    table.loc[i, "Флаг"] = 1
                    break

            if len(doc_table["Раздел"][j]) == 2 and (
                set(doc_table["Раздел"][j]) == set(first_two)
            ):
                words2 = set(doc_table["Название"][j])
                intersection_count = len(set(key_words).intersection(words2))
                if (
                    intersection_count != 0
                    and max_intersection_count < intersection_count
                ):
                    max_intersection_count = intersection_count
                    table.loc[i, "Флаг"] = 1
                    break

    if table["Флаг"][i] != 1:
        comments.append(table["Номер"][i])

def str2date(string, f_morph):
    """
        Получить дату из строки "Крайние даты" таблицы описи
    """
    string = string.strip()
    dashes_pattern = r'[-\u2010-\u2015\u2212\u2E3A\u2E3B]' #замена тире
    string = re.sub(dashes_pattern, '-', string)
    
    months = [ "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
    f_has_rus_mount = lambda s: any([True if m in s else False for m in months])
    has_dash = True if '-' in string else False
    
    has_rus_mount = f_has_rus_mount(string)
    # print(has_rus_mount, has_dash)
    if len(string) == 4 and len(re.findall(r"\d+", string)) == 1: #только year
        return pd.to_datetime(string).to_pydatetime() #year
    
    if len(string) > 4 and not(has_dash) and not(has_rus_mount) and not("/" in string): #dd.mm.year
        date_pattern = r'\b\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})\b'
        match = re.search(date_pattern, string)
        return dateparser.parse(string, languages=['ru'])  #dd.mm.year
     
    if "-" in string and not(has_rus_mount): #dd.mm.year-dd.mm.year
        # date_pattern = r'(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})'
        date_pattern = r'(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})'
        string = string.replace(" ","")
        matches = re.findall(date_pattern, string)
        
        if matches:
            dates = []
            for match in matches:
                day, month, year = match
                date_obj = datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
                dates.append(date_obj)
            if len(dates) == 1:
                return dates[0], None #<проблема
            return dates[0], dates[1] #dd.mm.year-dd.mm.year
        
        date_pattern = r"\s*(\d{4})\s*-\s*(\d{4})\s*"
        match = re.search(date_pattern, string)
        if match:
            start_year, end_year = match.groups()
            return datetime.strptime(f"{start_year}", "%Y"), datetime.strptime(f"{end_year}", "%Y") # year-year
        
    if has_rus_mount: #русские месяцы
        months = {
                "январь": 1,
                "февраль": 2,
                "март": 3,
                "апрель": 4,
                "май": 5,
                "июнь": 6,
                "июль": 7,
                "август": 8,
                "сентябрь": 9,
                "октябрь": 10,
                "ноябрь": 11,
                "декабрь": 12,
            }
        if "-" in string:
            date_pattern = r'([а-яА-Я]+)\s*-\s*([а-яА-Я]+)\s*(\d{4})'
            match = re.search(date_pattern, string)

            if match:
                start_month, end_month, year = match.groups()
                start_num = months.get(f_morph.parse(start_month.lower())[0].normal_form)
                # start_num = months.get(start_month.lower())
                # end_num = months.get(end_month.lower())
                end_num = months.get(f_morph.parse(end_month.lower())[0].normal_form)
                year = int(year)
                last_day = calendar.monthrange(year, end_num)[1]
                return datetime(year = year, month=start_num, day = 1), datetime(year = year, month=end_num, day = last_day) # mount-mount year
        else:
            date_pattern = r'([а-яА-Я]+)\s*(\d{4})'
            match = re.search(date_pattern, string)
            if match:
                start_month, year = match.groups()
                
                
                start_num = months.get(f_morph.parse(start_month.lower())[0].normal_form)
                year = int(year)
                return  datetime(year = year, month=start_num, day = 1), datetime(year = year, month=start_num, day = 31) # mount year
    if "/" in string:
        year_pattern = r'\s*(\d{4})\s*/\s*(\d{4})\s*'
        match = re.search(year_pattern, string)
        if match:
            year1, year2 = match.groups()
        return  datetime(year = int(year1), month=1, day = 1), datetime(year = int(year2), month=12, day = 31) # mount year
    
def parse_header(text, f_morph):
    """
    Получить дату из заголовка описи дел
    """
    text = text.strip()
    dashes_pattern = r"[-\u2010-\u2015\u2212\u2E3A\u2E3B]"  # замена тире
    text = re.sub(dashes_pattern, "-", text)
    months = {
        "январь": 1,
        "февраль": 2,
        "март": 3,
        "апрель": 4,
        "май": 5,
        "июнь": 6,
        "июль": 7,
        "август": 8,
        "сентябрь": 9,
        "октябрь": 10,
        "ноябрь": 11,
        "декабрь": 12,
    }
    if "-" in text:
        pattern = r"([а-яА-Я]+)\s*-\s*([а-яА-Я]+)\s*(\d{4})"
        match = re.search(pattern, text)
        if match:
            start_month, end_month, year = match.groups()
            start_num = months.get(f_morph.parse(start_month.lower())[0].normal_form)
            # start_num = months.get(start_month.lower())
            # end_num = months.get(end_month.lower())
            end_num = months.get(f_morph.parse(end_month.lower())[0].normal_form)
            year = int(year)
            if start_num is None or end_num is None:
                return None
            last_day = calendar.monthrange(year, end_num)[1]
            return datetime(year=year, month=start_num, day=1), datetime(
                year=year, month=end_num, day=last_day
            )  # mount-mount year
    # за сентрябрь 2014 года
    if "год" in text:
        match = re.search(r"(?:за)\s*([а-яА-Я]+)\s*(\d{4})", text)
        if match:
            m, year = match.groups()
            mount = months.get(f_morph.parse(m.lower())[0].normal_form)
            if mount in range(1, 13):
                year = int(year)
                last_day = calendar.monthrange(year, mount)[1]
                return datetime(year=year, month=mount, day=1), datetime(
                    year=year, month=mount, day=last_day
                )

    match_year_b = re.search("\d{4} г.р.", text)
    if match_year_b:
        return None

    match = re.search("(?:за|на|в)\s*\d{4}\s*год", text)  # за 2014 год
    if match:
        year = int(re.search("\d{4}", match.string).group())
        return datetime(year=year, month=1, day=1)

    match = re.search(
        "(?:за|на|в)\s*\d{4}\s*-\s*\d{4}\s*год", text
    )  # за 2014-2024 годы
    if match:
        return str2date(match.string, f_morph)
    match = re.search("от\s*(\d{1,2})\s*([а-яА-Я]+)\s*(\d{4})\sгод", text)
    if match:  # от 15 сентября 2024
        d, m, y = match.groups()
        mount = months.get(f_morph.parse(m.lower())[0].normal_form)
        return datetime(year=int(y), month=mount, day=int(d))
    if "учебный" in text and "/" in text:
        y1, y2 = re.search(r"(\d{4})\s*/\s*(\d{4})", text).groups()
        y1 = int(y1)
        y2 = int(y2)
        last_day = calendar.monthrange(y2, 12)[1]
        return datetime(year=y1, month=1, day=1), datetime(
            year=y2, month=12, day=last_day
        )


def is_valid_data_header(text, f_morph):
    """
    Проверка дат в заголовке дела описи
    """
    months = {
        "январь": 1,
        "февраль": 2,
        "март": 3,
        "апрель": 4,
        "май": 5,
        "июнь": 6,
        "июль": 7,
        "август": 8,
        "сентябрь": 9,
        "октябрь": 10,
        "ноябрь": 11,
        "декабрь": 12,
    }

    text = text.strip()
    dashes_pattern = r"[-\u2010-\u2015\u2212\u2E3A\u2E3B]"  # замена тире
    text = re.sub(dashes_pattern, "-", text)

    if "-" in text:
        pattern = r"([а-яА-Я]+)\s*-\s*([а-яА-Я]+)\s*(\d{4})"
        match = re.search(pattern, text)
        if match:
            start_month, end_month, year = match.groups()
            start_num = months.get(f_morph.parse(start_month.lower())[0].normal_form)
            end_num = months.get(f_morph.parse(end_month.lower())[0].normal_form)
            if start_num == 1 and end_num == 12:
                return False

        pattern = "\s*([а-яА-Я]+)\s*(\d{4})\s*-\s*([а-яА-Я]+)\s*(\d{4})"
        match = re.search(pattern, text)
        if match:
            start_month, start_year, end_month, end_year = match.groups()
            start_num = months.get(f_morph.parse(start_month.lower())[0].normal_form)
            end_num = months.get(f_morph.parse(end_month.lower())[0].normal_form)
            if start_num == 1 and end_num == 12:
                return False
    return True


def is_last_date(text):
    date_pattern = r"(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})"
    text = text.replace(" ", "")
    matches = re.findall(date_pattern, text)

    if matches:
        dates = []
        for match in matches:
            day, month, year = match
            date_obj = datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
            dates.append(date_obj)
        if len(dates) != 2:
            return False
        d1 = dates[0]
        d2 = dates[1]
        if d1.day == 1 and d1.month == 1 and d2.day == 31 and d2.month == 12:
            return True
    return False
