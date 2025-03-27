from src.app.checker.utils.text_utils import find_number_page
from src.app.checker.utils.text_utils import *
from src.app.checker.utils.table_utils import (
    get_vertical_lines,
    get_tocken,
    drop_table_lines,
    count_board_line_in_page,
    get_begin_page,
)
from dateutil.relativedelta import relativedelta
from src.app.checker.utils.file_utils import timer, CustomLRUCache
from src.app.checker.const import END_PAGE_PATTERN, END_PAGE_PATTERN2
import datetime
from datetime import date
from datetime import datetime
import pdfplumber
import re
import string
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
import functools as f
import os
import yaml
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

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


segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)

names_checks = {
    0: "Проверка №0 : Шаблон таблицы",
    1: "Проверка №1 : Название архива на титульном листе описи соответствует\n перечню архивных органов и учреждений Челябинской области",
    2: "Проверка №2 : Название фонда на титульном листе описи соответствует\n названию организации по СП1 или названию фонда по СП3",
    3: "Проверка №3 : Номер фонда на титульном листе\n соответствует номеру фонда по СП4",
    4: "Проверка №4 : Название фонда на титульном листе описи\n соответствует названию фонда на первом листе описи (перед таблицей)",
    5: "Проверка №5 : Номер фонда на титульном листе соответствует\n номеру фонда на первом листе описи (перед таблицей)",
    6: "Проверка №6 : Крайние даты описи на титульном листе\n соответствуют крайним датам на первом листе описи (перед таблицей)",
    7: "Проверка №7 : Крайние даты единиц хранения (строки в таблице) в графе 4 «крайние даты»\n соответствуют диапазону крайних дат на титульном листе описи, на первом листе описи",
    8: "Проверка №8 : Название организации - источника комплектования в блоке\n со словом УТВЕРЖДЕНО соответствует названию организации - источника комплектования в блоке со словами СОГЛАСОВАНО ЭК",
    9: "Проверка №9 : Номер описи на титульном листе соответствует номеру описи на первом листе описи (перед таблицей)",
    10: "Проверка №10: Проверка заголовка дела в описи на соответствие\n типовому перечню  и по сроку хранения дела",
    11: "Проверка №11: Проверка графы «крайние даты» и графы «срок хранения»\n по описи дел по личному составу на соответствие Правилам",
    12: "Проверка №12: Проверка описи постоянного хранения по отраслевым перечням утверждённым на ЭПК",
    13: "Проверка №13: Проверка соответствия номеров дел — порядковому расположению",
    14: "Проверка №14: Проверка графы 5 описи «Кол-во листов» - не превышает 250 листов",
    15: "Проверка №15: Проверка итоговой записи в описи на соответствие количеству дел в описи",
    16: "Проверка №16: Проверка итоговой записи в описи на соответствие номерам дел в описи",
    17: "Проверка №17: Проверка даты утверждения описи в блоке «УТВЕРЖДЕНО»",
    18: "Проверка №18: Проверка графы «Срок хранения» в описи дел по личному составу",
    19: "Проверка №19: Проверка описи постоянного хранения по отраслевым перечням утверждённым на ЭПК (выборы)",
    20: "Проверка №20: Проверка описей дел постоянного хранения комитетов/управлений/отделов архитектуры\n и градостроительства, Научно-исследовательских институтов (НИИ) и описей Научно-технической документации",
    21: "Проверка №21: Проверка присутствия грифа согласования описи с экспертной комиссией организации под описью",
    22: "Проверка №22: Проверка соответствия крайних дат в заголовке дела крайним датам в графе «крайние даты» описи",
    23: "Проверка №23: Проверка наличия нумерации документов в деле в заголовках дел распорядительных документов",
    24: "Проверка №24: Проверка отсутствия ошибок в формулировке крайних дат в описи",
    25: "Проверка №25: Проверка наличия списка сокращений в описи дел",
    26: "Проверка №26: Проверка наличия в описи «предисловия» или «дополнения к предисловию» \n состоящего из «1. История фондообразователя» и «2. История фонда»",
    27: "Проверка №27: Проверка наличия грифа подписи после основного текста «предисловия» или «дополнения к предисловию» в описи",
    28: "Проверка №28: Проверка предисловия к описи на соответствие диапазону\n крайних дат на титульном листе описи, на первом листе описи",
}


@CustomLRUCache(maxsize=15000)
def get_embedding(model, text):
    return model.encode(text.lower(), convert_to_tensor=True)


with open(os.getcwd() + "/src/app/config.yaml") as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)



if config["local"]:
    model = SentenceTransformer(
        "/root/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9"
    )
else:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@timer
def check_1(dict_text, sp2):
    """

    Название архива на титульном листе описи соответствует перечню
    архивных органов и учреждений Челябинской области
    """
    text_first_page = dict_text[0]

    mod_text_first_page = ''
    for symbol in ["\n", " "]:
        mod_text_first_page = clear_text(text_first_page, symbol)

    if any(
        list(
            map(
                lambda x: " ".join(x.split(" ")).lower().strip()
                in mod_text_first_page.lower(),
                sp2['Организация'],
            )
        )
    ):
        return True, ""

    name_archieve = list(filter(lambda x: x != " ", text_first_page.split("\n")))[
        0
    ].strip()
    sp2_df = pd.DataFrame()

    sp2_df["archieve"] = sp2
    sp2_df["name_archieve"] = [get_tocken(name_archieve)] * len(sp2_df)
    name_archieve_emb = get_embedding(model, get_tocken(name_archieve))

    def check_func(model, text):
        text_tocken = get_tocken(text)
        text_emb = get_embedding(model, text_tocken)
        pred = get_similar_2(model, text_emb, name_archieve_emb).item()
        return pred > 0.98

    # print(get_embedding.cache_size())
    sp2_df["check"] = sp2_df["archieve"].apply(lambda text: check_func(model, text))

    return any(sp2_df["check"]), ""


@timer
def check_2_search(text, page_blocks, sp1, sp3):
    """
    Поиск по бд
    """

    def get_main_words(name_fond, word_counts):
        if len(name_fond) > 2:
            words = name_fond.split(" ")
            counts = {}
            for word in words:
                word = word.lower()
                if word in word_counts:
                    if word_counts[word] > 3:
                        counts[word] = word_counts[word]
            if len(counts) < 2:
                counts[words[0]] = 1
                counts[words[1]] = 1
            sorted_items_by_value = dict(
                sorted(counts.items(), key=lambda item: item[1])
            )

            a, b = list(sorted_items_by_value.keys())[:2]
        return a, b, sorted_items_by_value

    frame_counts = pd.concat((sp1.iloc[:, 0], sp3.iloc[:, 1]))
    all_words = {}
    # print(frame_counts.shape)
    # return frame_counts
    for row in frame_counts:
        words = row.lower().split(" ")
        for word in words:
            res = re.findall("[a-zA-Zа-яА-ЯёЁ]+", word)
            if len(res) > 0:
                word = res[0]
            else:
                continue
            if word in all_words:
                all_words[word] += 1
            else:
                all_words[word] = 1

    name_fond = get_name_fond(text, page_blocks)
    if (
        name_fond == "ERROR: В начале первой странице не найден отдел архива"
        or len(name_fond) == 0
    ):
        return False, name_fond
    elif name_fond == "ERROR: Имя фонда не найдено":
        return False, name_fond

    a, b, ws = get_main_words(name_fond, all_words)

    name_fond = name_fond.lower()
    mask1 = frame_counts.str.lower().str.contains(a)
    mask2 = frame_counts.str.lower().str.contains(b)
    query = pd.concat((frame_counts[mask1], frame_counts[mask2]))

    # print("ИМЯ ФОНДА = ", name_fond, query.shape)
    name_fond_tok = get_tocken(name_fond)
    name_fond_emb = get_embedding(model, name_fond_tok)
    sp1_names_fond_tok = list(map(lambda x: get_tocken(x), query.to_list()))
    sp1_names_fond_emb = list(
        map(lambda x: get_embedding(model, x), sp1_names_fond_tok)
    )

    fond_set = set(name_fond.lower().split(" "))
    result_orgs = {}
    for x_fond, fond in zip(sp1_names_fond_emb, sp1_names_fond_tok):
        r = get_similar_2(model, x_fond, name_fond_emb)
        s1 = set(fond.lower().split(" "))
        if r > 0.8:
            result_orgs[r.item()] = {"f": fond, "c": len(s1.intersection(fond_set))}
        if r > 0.995:
            return True, f"Фонд описи '{name_fond}' найден в списке: '{fond}'"
    result_orgs = dict(
        sorted(result_orgs.items(), key=lambda item: item[0], reverse=True)
    )
    result_orgs = list(result_orgs.items())
    if len(result_orgs) == 0:
        return False, f"{name_fond} - фонд не найден"
    else:
        max_count = 0
        candidates = []
        for x in result_orgs:
            if max_count <= x[1]["c"]:
                candidates.append(x)
                max_count = x[1]["c"]
        comment = ""
        for x in candidates[:3]:
            c = f"{x[1]['f']}, соотвествие = {round(x[0],3)}, совпадений слов {x[1]['c']}; "
            comment += c
        return True, f"Наиболее подходящие для '{name_fond}' фонды: {comment}"


# @timer
# def check_2(text, sp1, sp3):
#     """
#     Название фонда на титульном листе описи соответствует названию организации по
#     СП1 или названию фонда по СП3
#     """
#     text_first_page = text[0]

#     name_archieve = " ".join(
#         list(filter(lambda x: x != " ", text_first_page.split("\n")))[0]
#         .lower()
#         .strip()
#         .split()
#     )

#     for symbol in ["\n", " "]:
#         text_first_page = clear_text(text_first_page, symbol)

#     name_fond = get_name_fond(text).lower()
#     if name_fond == "ERROR: В начале первой странице не найден отдел архива":
#         return False, name_fond

#     number_fond = get_number_fond(text, 0)

#     sp1 = sp1[sp1.iloc[:, 1].str.lower().str.strip().str.match(name_archieve)]
#     sp3 = sp3[sp3.iloc[:, 0].str.lower().str.strip().str.match(name_archieve)]

#     sp1_names_fond = list(
#         map(
#             lambda x: remove_punctuation(clear_text(x, "\n")),
#             sp1.iloc[:, 0].str.lower().str.strip().fillna(""),
#         )
#     )
#     sp3_names_fond = list(
#         map(
#             lambda x: remove_punctuation(clear_text(x, "\n")),
#             sp3.iloc[:, 1].str.lower().str.strip().fillna(""),
#         )
#     )

#     name_archieve_pos = text_first_page.lower().find("архив") + 1
#     name_fond_pos = remove_punctuation(text_first_page).lower().find(name_fond) + 1
#     number_fond_pos = text_first_page.lower().find(str(number_fond)) + 1

#     check_pos = not (name_archieve_pos < name_fond_pos < number_fond_pos)
#     if check_pos and name_archieve_pos and name_fond_pos and number_fond_pos:
#         return False, ""

#     result_1 = any(
#         list(map(lambda x: name_fond in " ".join(x.lower().split(" ")), sp1_names_fond))
#     )
#     result_2 = any(
#         list(map(lambda x: name_fond in " ".join(x.lower().split(" ")), sp3_names_fond))
#     )

#     if result_1 or result_2:
#         return True, "Абсолютное совпадение"


#     # если нет абсолютного совпадения ищем через косинус

#     name_fond_tok = get_tocken(name_fond)
#     name_fond_emb = get_embedding(model, name_fond_tok)

#     sp1_names_fond_tok = list(map(lambda x: get_tocken(x), sp1_names_fond))
#     sp1_names_fond_emb = list(
#         map(lambda x: get_embedding(model, x), sp1_names_fond_tok)
#     )
#     sp1_pred = any(
#         list(
#             map(
#                 lambda x: get_similar_2(model, name_fond_emb, x).item() > 0.96,
#                 sp1_names_fond_emb,
#             )
#         )
#     )

#     sp3_names_fond_tok = list(map(lambda x: get_tocken(x), sp3_names_fond))
#     sp3_names_fond_emb = list(
#         map(lambda x: get_embedding(model, x), sp3_names_fond_tok)
#     )
#     sp3_pred = any(
#         list(
#             map(
#                 lambda x: get_similar_2(model, name_fond_emb, x).item() > 0.96,
#                 sp3_names_fond_emb,
#             )
#         )
#     )

#     return sp1_pred or sp3_pred, ""

@timer
def check_2(text, sp1, sp3):

    def extract_caps_and_following_text(text, keywords):
        lines = text.split("\n")
        extracted_lines = []
        capturing = False
        
        # Преобразуем ключевые слова в нижний регистр для проверки
        keywords_lower = [kw.lower() for kw in keywords]
        
        for line in lines:
            stripped_line = line.strip()
            
            if not capturing and stripped_line.isupper():  # Проверяем, написана ли строка капсом
                capturing = True
            
            if capturing:
                # Проверяем, содержит ли строка одно из ключевых слов
                if any(keyword in stripped_line.lower() for keyword in keywords_lower):
                    break
                extracted_lines.append(line)
        
        return "\n".join(extracted_lines).strip()
    name_fond = extract_caps_and_following_text(text[0], ["оаф", "опись", "фонд"])

    if name_fond == '':
        return True, 'В начале первой страницы не найдено название фонда'

    lem_fond = preprocess_and_tokenize(name_fond)

    # sp1['Фонд'] = sp1['Наименование организации'] + ' '+ sp1['Архивный отдел']
    sp1['Фонд'] = sp1['Наименование организации']
    # sp3['Фонд'] = sp3['Архив'] + ' '+ sp3['Наименование'] 
    sp3['Фонд'] = sp3['Наименование']   

    def process_sp1_sp3(df, lem_fond):
        df['Фонд_Лемма'] = df['Фонд'].apply(preprocess_and_tokenize)
        df["intersection_count"] = df['Фонд_Лемма'].apply(lambda x: len(lem_fond & x) / len(lem_fond))
        top_3 = df.nlargest(3, "intersection_count")["Фонд"].tolist()
        # filtered_df = df[df["intersection_count"] > 0.5]
        # top_3 = filtered_df.nlargest(3, "intersection_count")["Фонд"].tolist() 
        return top_3
    
    lis1= process_sp1_sp3(sp1, lem_fond)
    lis3 = process_sp1_sp3(sp3, lem_fond)

    if lis1 and lis3:
        return True, f'Найдены близкие элементы из СП1 {lis1} и СП3 {lis3}'
    elif lis1:
        return True, f'Найдены близкие элементы из СП1 {lis1}'
    elif lis3:
        return True, f'Найдены близкие элементы из СП3 {lis3}'
    else: return False, 'Не найдено совпадений'


# Проверка 3
@timer
def check_3(text, page_blocks, sp4):
    """
    Название архива на титульном листе описи соответствует перечню
    архивных органов и учреждений Челябинской области
    """
    text_first_page = text[0]

    name_archieve_1 = (
        list(filter(lambda x: x != " ", text_first_page.split("\n")))[0].lower().strip()
    )
    name_archieve_2 = "".join(name_archieve_1.lower().strip().split(" "))

    for symbol in ["\n", " "]:
        text_first_page = clear_text(text_first_page, symbol)

    name_fond = get_name_fond(text, page_blocks)

    name_fond_check_1 = "".join(name_fond.lower().strip().split(" "))
    number_fond = get_number_fond(text, 0)

    sp4_2 = sp4.copy()

    sp4["tokens"] = sp4.iloc[:, 0].apply(lambda x: get_tocken(x.lower()))
    sp4_2["tokens"] = sp4_2.iloc[:, 0].apply(lambda x: get_tocken(x.lower()))

    sp4.iloc[:, 1] = sp4.iloc[:, 1].apply(
        lambda x: "".join(x.lower().strip().split(" "))
    )
    sp4.iloc[:, 0] = sp4.iloc[:, 0].apply(
        lambda x: remove_punctuation("".join(x.lower().strip().split(" ")))
    )

    sp4['similar_simbols'] = sp4.iloc[:, 0].apply(lambda x: get_similar_3(x, name_fond_check_1))

    sp4_candidates = sp4[sp4['similar_simbols'] > 75].copy()
    sp4_number_fonds_2 = sp4_candidates[
        sp4_candidates['similar_simbols'] == sp4_candidates['similar_simbols'].max()
    ]

    if not sp4_number_fonds_2.empty:
        sp4_number_fonds_2 = sp4_number_fonds_2.iloc[:, 2].to_list()
    else:
        sp4_number_fonds_2 = []

    sp4 = sp4[sp4.iloc[:, 1].str.lower().str.strip().str.match(name_archieve_2)]

    sp4_number_fonds_1 = (
        sp4[sp4.iloc[:, 0].str.contains(name_fond_check_1)].iloc[:, 2].to_list()
    )

    if (number_fond in sp4_number_fonds_1) or (number_fond in sp4_number_fonds_2):
        return True

    name_fond_tok = get_tocken(name_fond.lower())
    name_fond_emb = get_embedding(model, name_fond_tok)

    sp4_2["embs"] = sp4_2["tokens"].apply(lambda x: get_embedding(model, x))
    sp4_2["pred"] = sp4_2["embs"].apply(
        lambda x: get_similar_2(model, name_fond_emb, x).item()
    )

    sp4_number_fonds = sp4_2[sp4_2["pred"] > 0.923].iloc[:, 2].to_list()

    return number_fond in sp4_number_fonds


@timer
def check_4(text, page_blocks):
    """
    Название фонда на титульном листе описи соответствует
    названию фонда на первом листе описи (перед таблицей)
    """
    name_fond = get_name_fond(text, page_blocks)
    if name_fond[:5] == "ERROR":
        return False, name_fond[6:]
    name_fond_table = get_name_fond_table(text, find_number_page_table(text))

    text_page = text[find_number_page(text, "утверждаю")]
    for symbol in ["\n", " "]:
        text_page = clear_text(text_page, symbol)

    if name_fond.lower() in remove_punctuation(text_page).lower():
        return True, ""

    name_fond_token = get_tocken(name_fond)
    name_fond_table_token = get_tocken(name_fond_table)

    name_fond_emb = get_embedding(model, name_fond_token)
    name_fond_table_emb = get_embedding(model, name_fond_table_token)

    pred = get_similar_2(model, name_fond_emb, name_fond_table_emb)
    return pred.item() > 0.88, f"Соответствие {round((pred*100).item(),2)}%"


@timer
def check_5(text):
    """
    Номер фонда на титульном листе описи соответствует
    номеру фонда на первом листе описи (перед таблицей)
    """
    number_pg = find_number_page(text, "утверждаю")
    number_fond_main = get_number_fond(text, 0)
    number_fond_table = get_number_fond(text, number_pg)
    if (number_fond_main is None) or (number_fond_table is None):
        return False
    return number_fond_main == number_fond_table


@timer
def check_6(text):
    pr_text_0 = text[0]
    pr_text_0 = pr_text_0[pr_text_0.lower().find("опись") :]

    pr_text_1 = text[find_number_page(text, "утверждаю")]
    pr_text_1 = pr_text_1[pr_text_1.lower().find("опись") :]
    try:
        s = re.findall("за .{0,100}год", text[0])
        if len(s) > 0:
            if "-е" in s[0]:
                s1, s2 = re.findall("\d{4}", s[0])
                d1 = to_date(s1)[0]
                d2 = to_date(s2)[0]
            elif "/" in s[0]:
                s1, s2 = re.findall("\d{4}", s[0])
                d1 = to_date(s1)[0]
                d2 = to_date(s2)[0]
                d2 += relativedelta(years=1)
            else:
                d1, d2 = to_date(s[0][2:-3])

            d_next_1, d_next_2 = get_date_inventory(text)
            if d1.year == d_next_1.year and d2.year == d_next_2.year:
                return True


        # если диапазон дат через тире
        dates_page_0 = re.findall(
            "\s*\d{4}\s*-\s*\d{4}\s",
            pr_text_0.replace("–", "-").replace("--", "-").lower(),
        )
        if len(dates_page_0) > 0:
            dates_0 = to_date(dates_page_0[0])
            dates_1 = get_date_inventory(text)
            if (
                dates_0[0].year == dates_1[0].year
                and dates_0[1].year == dates_1[1].year
            ):
                return True
    except:
        pass
    date_list_0 = list(dates_extractor(pr_text_0))
    date_list_1 = list(dates_extractor(pr_text_1))

    if len(date_list_0) == 1:
        pr_text_0_date = get_correct_date(date_list_0[0])
        pr_text_1_date = get_correct_date(date_list_1[0])

        is_similar_dates = pr_text_0_date == pr_text_1_date
    else:
        pr_text_0_dates = list(map(lambda x: get_correct_date(x), date_list_0[:2]))
        pr_text_1_dates = list(map(lambda x: get_correct_date(x), date_list_1[:2]))
        is_similar_dates = pr_text_0_dates == pr_text_1_dates

    return is_similar_dates


@timer
def check_7(text, table):
    years = []
    try:
        s = re.findall("за .{0,100}год", text[0])
        if len(s) > 0:
            if "-е" in s[0]:
                s1, s2 = re.findall("\d{4}", s[0])
                d1 = to_date(s1)[0]
                d2 = to_date(s2)[0]
            elif "/" in s[0]:
                s1, s2 = re.findall("\d{4}", s[0])
                d1 = to_date(s1)[0]
                d2 = to_date(s2)[0]
            else:
                d1, d2 = to_date(s[0][2:-3])
            years = [int(d1.year), int(d2.year)]
    except:
        pr_text_1_date = list(dates_extractor(text[0][text[0].lower().find("опись") :]))
        if not pr_text_1_date:
            years_re = {
                0: [re.findall(r"\d{4}/\d{4}", text[0]), "/"],
                1: [re.findall(r"\d{4}-\d{4}", text[0]), "-"],
            }
            val_years_re = {key: val for key, val in years_re.items() if val[0] != []}
            if val_years_re:
                val_str = years_re[list(val_years_re.keys())[0]][0][0]
                val_split = years_re[list(val_years_re.keys())[0]][1]
                years = val_str.split(val_split)
        if len(pr_text_1_date) == 1:
            years = [int(pr_text_1_date[0].fact.year), int(pr_text_1_date[0].fact.year)]
        elif len(pr_text_1_date) > 1:
            years = [int(pr_text_1_date[0].fact.year), int(pr_text_1_date[1].fact.year)]

        if not years:
            return False

    years.sort()

    def check_value_row(text):
        nums = re.findall(r"\d+", text)
        filtered = list(filter(lambda x: (len(x) > 3), nums))
        if (len(filtered) == 1) and (len(nums) > 3):
            return False
        if "*" in text:
            return True
        return all(
            map(lambda x: (int(years[0]) <= int(x) <= int(years[1])), filtered)
        )

    return all(map(check_value_row, table.iloc[:, 3]))


@timer
def check_8(text):

    def extract_names(text):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        names = []
        for span in doc.spans:
            if span.type == 'PER':  
                span.normalize(morph_vocab)
                names.append((span.text, span.start, span.stop))  

        return names

    comment = ' '
    try:
        page_0_index = find_number_page_dict(text, 'утверждаю')
        pr_text_0 = text[page_0_index]

        text_temp = {k: text[k] for k in list(text.keys())[list(text.keys()).index(page_0_index):]}
        page_1_index = find_number_page_dict(text_temp, 'согласовано')
        pr_text_1 = text_temp[page_1_index]

    except KeyError: 
       return False, 'Отсутствие ключевого слова Утверждаю/Согласовано'

    start_index_0 = pr_text_0.lower().find('утверждаю')
    names = extract_names(pr_text_0)
    first_name = names[0]  
    snippet_0 = pr_text_0[start_index_0:first_name[1]]


    year_match_0 = re.search(r'\b(\d{4})\b', snippet_0)
    if year_match_0 is None:
      year_match_0 = re.search(r'[\s_]*(\d{4})\b', snippet_0)

    if year_match_0:
        year_index = year_match_0.start(1)
        end_index_0 = year_match_0.end(1)
        organization_0 = snippet_0[:end_index_0]
    else:
        organization_0 = snippet_0
    organization_1 = pr_text_1[pr_text_1.lower().find('согласовано'):pr_text_1.lower().find('согласовано')+150]
    for symbol in ['\n', ' ']:
        organization_0 = clear_text(organization_0, symbol)
        organization_1 = clear_text(organization_1, symbol)
 
    organization_0 = delete_data(organization_0, fio_ind=True)
    organization_1 = delete_data(organization_1, fio_ind=False)

    organization_0 = organization_0.replace('УТВЕРЖДАЮ', '').title()
    organization_1 = organization_1.replace('СОГЛАСОВАНО Протокол ЭК', '').replace('Протокол', '').replace('СОГЛАСОВАНО', '').replace('Согласовано', '').replace('ЭК', '').title()

    organization_1 = organization_1.replace("от", "")
    if not re.search(r"[А-Яа-яЁё]", organization_1):
       return False, 'Проверяемый блок должен находиться на одной странице'
    result, org0, org1 = check_sim(organization_0,organization_1)
    if (org0 == set() or org1 == set()) and result == False:
        return True, comment

    if result == False:
      is_substring = organization_0 in organization_1
      return is_substring, comment
    return result, comment


@timer
def check_9(text):
    pg_table = find_number_page(text, "утверждаю")


    number_opice_0, type_opice_0 = get_info_opice(text, 0)
    number_opice_1, type_opice_1 = get_info_opice(text, pg_table)

    type_opice_0_token = get_tocken(type_opice_0)
    type_opice_1_token = get_tocken(type_opice_1)

    type_opice_0_emb = get_embedding(model, type_opice_0_token)
    type_opice_1_emb = get_embedding(model, type_opice_1_token)

    res_check_type = get_similar_2(model, type_opice_0_emb, type_opice_1_emb).item()

    return (number_opice_0 == number_opice_1) and (res_check_type > 0.8)


@timer
def check_10(table, doc_table, dict_text, df):
    """
    Обрабатывает данные в таблице и устанавливает соответствие описи документам из СП5
    table - таблица описи
    doc_table - списки по таблицы СП5
    dict_text - текст описи
    df - таблица СП5

    Returns:
    ----------
    tuple
        - bool: True, если для всех документов в описи найдено соответствие в СП5, иначе False.
        - comments: Список комментариев о найденных совпадениях для каждого документа в описи.
    """
    # start_time = time.time()
    df.to_csv("sp5_save.csv")
    title = dict_text[0].lower()

    phrase1 = "дел постоянного хранения"
    phrase2 = "дел по личному составу"
    search_terms1 = [
        "комитет архитектуры",
        "отдел архитектуры",
        "градостроительства",
        "научно-исследовательский",
        "научно-технический",
        "научно-технической",
        "научно-исследовательский",
    ]
    search_terms2 = ["выбор", "избират"]

    if phrase1 in title:
        flag = 1
    elif phrase2 in title:
        flag = 0
    else:
        return True, [
            "Проверка только для описей по личному составу/постоянного хранения"
        ]

    if any(term in title for term in search_terms2):
        return True, ["Проверка данной описи (по выборам) проводится в 19 проверке"]

    if any(term in title for term in search_terms1):
        return True, ["Проверка данной описи (НТД) проводится в 20 проверке"]

    # Подготовка данных
    df = prep_sp5_table(df, flag)
    doc_table.rename(columns={doc_table.columns[1]: "Номер"}, inplace=True)
    doc_table["Номер"] = doc_table["Номер"].astype(str)
    doc_table["Раздел"] = doc_table["Раздел"].str.replace(
        r"^[^\W\d_]\)", "", regex=True
    )
    df["Номер"] = df["Номер"].astype(str)
    doc_table = doc_table.merge(df[df["contains_flag"]], on="Номер")
    table = prep_dataset_doc_10(table)
    table["Флаг"] = 0
    doc_table["Раздел"] = doc_table["Раздел"].apply(lambda x: x.lower().split())
    doc_table["Название"] = doc_table["Название"].apply(lambda x: x.lower().split())
    comments = []

    # Используем ThreadPoolExecutor для параллельной обработки строк
    with ThreadPoolExecutor(4) as executor:
        futures = []
        for i in range(len(table["Заголовок дела"])):
            futures.append(
                executor.submit(process_row_10, i, table, doc_table, comments)
            )

        # Ожидаем завершения всех задач
        for future in as_completed(futures):
            future.result()  # Чтобы ловить возможные исключения, если они были в процессе выполнения

    # print(f"Время выполнения поиска: {int((time.time() - start_time) // 60)} минут {(time.time() - start_time) % 60:.2f} секунд")

    all_matched = (table["Флаг"] == 1).all()
    end_comment = []
    if not all_matched:
        comments_str = ", ".join(map(str, comments))
        end_comment = [
            f"Запись(и) № {comments_str} не соответствует(ют) перечню из справочника СП5"
        ]

    return all_matched, end_comment


@timer
def check_11(text, table):
    """Проверка графы «крайние даты» и графы «срок хранения» по описи дел по личному составу на соответствие Правилам"""

    def check(row):
        """Вспомогательная функция для провеки наличия и корректности значений"""
        before_range = {"75 лет", "75 лет ЭПК"}  # диапазон с 2003 до year
        after_or_include_range = {"50 лет", "50 лет ЭПК"}
        try:
            numbers = list(
                row.iloc[3].strip().replace("-", ".").replace("–", ".").split(".")
            )
            year = int(numbers[-1].strip())
            if len(numbers) > 3:
                year = int(numbers[2].strip())

            if year < 2003:
                res = (
                    " ".join(
                        x.strip() for x in row.iloc[4].strip().split(" ") if x != ""
                    )
                    in before_range
                )
            else:
                res = (
                    " ".join(
                        x.strip() for x in row.iloc[4].strip().split(" ") if x != ""
                    )
                    in after_or_include_range
                )
        except:
            res = False
        return res

    if not "по личному составу" in text[0].split("№")[-1].lower():
        return True, "Не является описью дел по личному составу"

    tab = table.copy()

    tab = tab[(tab.iloc[:, 4] != "") & (tab.iloc[:, 3] != "")]

    tab["result"] = tab.apply(check, axis=1)
    result = int(tab["result"].sum()) == len(tab)

    return result, ""


@timer
def check_12(text, table, sp, name="default_error"):
    """Проверка описи постоянного хранения по отраслевым перечням утверждённым на ЭПК"""

    def check_func(model, text, t_list):
        """вспомогательная функция для вычисления схожести текстов"""
        for t in t_list:
            pred = get_similar_2(model, text, t).item()
            if pred > 0.63:
                return 1
        return 0

    dict_names = {
        7: "Перечень документов постоянного хранения, подлежащих включению в опись Администраций/Советов сельских поселений Челябинской области",
        8: "Перечень документов постоянного хранения, подлежащих включению в опись № 1 дел постоянного хранения Управления культуры администрации города/района",
        9: "Перечень документов постоянного хранения, подлежащих включению в опись № 1 дел постоянного хранения Управления образования администрации города/района",
        10: "Перечень документов постоянного хранения, подлежащих включению в опись № 1 дел постоянного хранения государственного бюджетного учреждения здравоохранения",
        11: "Перечень документов постоянного хранения, подлежащих включению в опись № 1 дел постоянного хранения управлений / отделов имущества администрации муниципального образования",
        12: "Примерный Перечень документов, подлежащих включению в опись №1 дел постоянного хранения Управления/Отдела по сельскому хозяйству администрации района",
        13: "Перечень документов постоянного хранения, подлежащих включению в опись дел постоянного хранения комитетов/управлений/отделов администраций муниципальных образований, осуществляющих экономическую деятельность",
        14: "Перечень документов, подлежащих включению в опись дел постоянного хранения управлений по физической культуре и спорту города/округа/района",
    }

    if not re.search("постоян.+хранен\w+", text[0].lower()):
        return True, "Не является описью постоянного хранения"

    tab = table.copy()


    pg_num = find_number_page(text, "утверждаю")
    organization_type = re.sub(
        "\s+", " ", text[pg_num].lower().split("опись")[0]
    ).lower()

    if re.search("выборам", organization_type):
        return True, "Проверка данной описи (по выборам) проводится в 19 проверке"

    spd = sp.copy()

    if bool(
        re.search(
            "(управление\s+культуры|отдел\s+культуры|комитет\s+по\s+культуре|управление\s+по\s+культуре)",
            organization_type,
        )
    ):
        num = 8
    elif bool(re.search("образован", organization_type)):
        num = 9
    elif bool(re.search("здравоохранен", organization_type)):
        num = 10
    elif bool(re.search("имуществ", organization_type)):
        num = 11
    elif bool(re.search("хозяйств", organization_type)):
        num = 12
    elif bool(re.search("экономичес|экономик", organization_type)):
        num = 13
    elif bool(
        re.search(
            "(культур.+спорт|спорт.+культур|физкультур|физическ.+культур)",
            organization_type,
        )
    ):
        num = 14
    elif bool(
        re.search(
            "(самоуправлен|администрац|собран|совет)[а-я ]*(поселен|город)",
            organization_type,
        )
    ):
        num = 7
    else:
        return True, "Нет соответствующего заголовка из перечня"

    spd = spd[spd.iloc[:, 1].apply(str) == str(num)]

    tab["tocken"] = tab.iloc[:, 2].apply(lambda t: get_tocken(t, True))
    spd["emb"] = spd["tocken"].apply(lambda text: get_embedding(model, text))
    tab["emb"] = tab["tocken"].apply(lambda text: get_embedding(model, text))

    tab["check"] = tab["emb"].apply(lambda text: check_func(model, text, spd["emb"]))

    errors = tab[tab["check"] == 0].iloc[:, 0].tolist()
    comment = f'Для проверки использовался справочник СП-{num} "{dict_names[num]}".'
    if errors:
        comment += (
            " Записи №: "
            + ", ".join([x for x in errors if x != ""])
            + " не соответствуют перечню из справочника"
        )
    return int(tab["check"].sum()) / len(tab) >= 0.7, comment


@timer
def check_17(text):
    current_year = date.today().year
    pr_text = text[find_number_page(text, "утверждаю")]
    dates = list(
        dates_extractor(
            pr_text[
                pr_text.lower().find("утверждаю") : pr_text.lower().find("утверждаю")
                + 300
            ]
        )
    )

    year = ""
    if len(dates) > 0:
        year = dates[0].fact.year

    return year == current_year


@timer
def check_13(table):
    """
    Проверка соответствия номеров дел — порядковому расположению
    """
    numbers = table.iloc[:, 0].values.astype(
        int
    )  # если индексы записаны как число np.int. А если "131а","131б"
    ans = False
    if all(numbers[1:] - numbers[:-1] == 1):
        ans = True
    return ans


@timer
def check_14(text, table):
    """ """
    etalon = 250
    if table.shape[1] == 6:
        index = 4
    elif table.shape[1] == 7 or table.shape[1] == 8:
        index = 5
    else:
        return (
            False,
            f"Неудалось! Индекс 'Количества листов' не найдет. В таблице {table.shape[1]} колонок",
        )

    col = table.columns[index]
    table[col] = table[col].fillna("")

    table[col] = table[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    table[col] = table[col].apply(lambda x: "".join(re.findall("\d+", x)))
    count_lists_diff = (
        table[(table[col] != "") & table[col].notnull()][col]
        .astype(int)
        .apply(lambda x: etalon - x)
    )

    min_values = min(count_lists_diff)

    if -20 <= min_values < 0:
        return True, "Предупреждение"
    elif min_values < 0:
        return False, ""

    return True, ""


@timer
def check_15_and_16(text, table):
    """
    Проверка итоговой записи в описи на соответствие количеству дел в описи
    Проверка итоговой записи в описи на соответствие номерам дел в описи
    """
    numbers = table.iloc[:, 0].values.astype(int)
    count_index_table = len(numbers)
    st_itable = numbers[0]
    end_itable = numbers[-1]

    number_pg = find_number_page(text, END_PAGE_PATTERN)
    ans = re.search("опис[-\w\d\D №()А-я]{1,100}внесен", text[number_pg].lower())
    if ans:
        s = ans.end()
    else:
        s = -1
    end = text[number_pg].lower().find("литер")
    sub_str = text[number_pg].lower()[s:end]
    nums = [int(x) for x in re.findall(r"\d+", sub_str)]
    if len(nums) == 3:
        count = nums[0]
        start = nums[1]
        end = nums[2]
    else:
        return False, False, f"Найденные №п/п {nums}"

    if count_index_table == count:
        if st_itable == start and end_itable == end:
            return True, True, ""
        else:
            return (
                True,
                False,
                f"В конце описи №п/п с {start} по {end} в таблице c {st_itable} по {end_itable}",
            )
    else:
        return False, False, ""


@timer
def check_18(text, table, cols):
    """Проверка графы «Срок хранения» в описи дел по личному составу"""

    def check(row):
        """Вспомогательная функция для провеки наличия и корректности значений"""
        befor_2002 = ["75 лет", "75 лет ЭПК"]
        after_2002 = ["50 лет", "50 лет ЭПК"]

        # if row['Крайние даты'].strip() == '' or row['Срок хранения'].strip() == '':
        #     return False
        try:
            year = min([int(x) for x in re.findall("\d{4}", row.iloc[3])])
            if year <= 2002:
                res = (
                    " ".join(
                        x.strip() for x in row.iloc[4].strip().split(" ") if x != ""
                    )
                    in befor_2002
                )
            else:
                res = (
                    " ".join(
                        x.strip() for x in row.iloc[4].strip().split(" ") if x != ""
                    )
                    in after_2002
                )
        except:
            res = False
        return res

    if not "по личному составу" in text[0].split("№")[-1].lower():
        return True, "Не является описью дел по личному составу"
    # if not list(filter(lambda x: "срокхранения" in x,[x.strip().lower().replace(" ", "") for x in cols],)):
    #     return True, 'Нет колонки "Срок хранения"'

    comment = ""
    tab = table.copy()

    tab = tab[(tab.iloc[:, 4] != "") & (tab.iloc[:, 3] != "") & (tab.iloc[:, 2] != "")]

    tab["result"] = tab.apply(check, axis=1)
    result = int(tab["result"].sum()) == len(tab)
    if not result:
        errors = tab[tab["result"] == False].iloc[:, 0].tolist()
        comment = "Ошибка в описях: " + ", ".join(errors)

    return result, comment


@timer
def check_19(table, dict_text, sps):
    """
    Обрабатывает данные в таблице и устанавливает соответствие описи документам из СП15-СП21

    table - таблица описи

    dict_text - текст описи

    Returns:
    ----------
    tuple
        - bool: True, если для всех документов в описи найдено соответствие в СП5, иначе False.
        - comments: Список комментариев о найденных совпадениях для каждого документа в описи.
    """

    def check_words_in_text(text):
        pattern_1 = r"\bвыбор\w*\b"
        pattern_2 = r"\bизбират\w*\b"
        match_1 = re.search(pattern_1, text, re.IGNORECASE)
        match_2 = re.search(pattern_2, text, re.IGNORECASE)
        return bool(match_1 or match_2)

    title = dict_text[0].lower()

    if check_words_in_text(title) == False:
        return True, ["Проверка только для описей по выборам"]

    num_doc = 0
    if find_flexible_word(title, "президента"):
        num_doc = 19
    if find_flexible_word(title, "губернатора челябинской области"):
        num_doc = 17
    if find_flexible_word(title, "изменений в конституцию"):
        num_doc = 15
    if find_flexible_word(title, "государственной думы федерального собрания"):
        num_doc = 16
    if find_flexible_word(title, "депутатов законодательного собрания"):
        num_doc = 18
    if (find_flexible_word(title, "выборам глав")) or (
        find_flexible_word(title, "выборам главы")
    ):
        num_doc = 21
    if (
        (find_flexible_word(title, "депутатов сельских поселений"))
        or (find_flexible_word(title, "депутатов городских поселений"))
        or (find_flexible_word(title, "депутатов собрания депутатов"))
    ):
        num_doc = 20

    if num_doc == 0:
        return False, ["Опись не соответствует справочникам СП15-СП21"]

    table = prep_table(table)
    doc_table = prep_sp(sps)

    comments = []
    indices_to_drop = doc_table[doc_table["номер_справочника"] != str(num_doc)].index
    doc_table = doc_table.drop(indices_to_drop)
    doc_table = doc_table.reset_index()

    for i in range(len(table["Заголовок дела"])):
        words1 = table["Заголовок дела"][i].lower().split()
        words1 = {word for word in words1 if not word.isdigit()}
        max_intersection_count = 0
        for j in range(len(doc_table["Заголовок дела"])):
            words2 = set(doc_table["Заголовок дела"][j].lower().split())
            intersection_count = len(set(words1).intersection(words2))
            if intersection_count > 2 and max_intersection_count < intersection_count:
                max_intersection_count = intersection_count
                table.at[i, "Флаг"] = 1
                break

            if (find_flexible_word(table["Заголовок дела"][i], "сводная таблица")) and (
                find_flexible_word(doc_table["Заголовок дела"][j], "сводная таблица")
            ):
                table.at[i, "Флаг"] = 1
                break

        if table["Флаг"][i] != 1:
            comments.append(table["№ п/п"][i])

    if bool((table["Флаг"] == 1).all()) == False:
        comments_str = ", ".join(map(str, comments))
        end_comment = [
            f"Запись(и) № {comments_str} не соответствует(ют) перечню из справочника СП{num_doc}"
        ]
    else:
        end_comment = [f"Проверка проведена по справочнику СП{num_doc}"]
    return bool((table["Флаг"] == 1).all()), end_comment


@timer
def check_20(table, doc_table, dict_text, df):
    """
    Обрабатывает данные в таблице и устанавливает соответствие описи документам из СП6
    table - таблица описи
    doc_table - СП6 список после таблиц
    dict_text - текст описи
    df - СП6 объединенные таблицы

    Returns:
    ----------
    tuple
        - bool: True, если для всех документов в описи найдено соответствие в СП6, иначе False.
        - comments: Список комментариев о найденных совпадениях для каждого документа в описи.
    """

    title = dict_text[0].lower()
    phrase1 = "постоянного хранения"
    search_terms1 = [
        "комитет архитектуры",
        "отдел архитектуры",
        "градостроительства" "научно-исследовательский",
        "научно-технический",
        "научно-технической",
        "научно-исследовательский",
    ]

    # start_time1 = time.time()
    if phrase1 not in title:
        return True, ["Проверка только для описей постоянного хранения"]

    if any(term in title for term in search_terms1) == False:
        return True, ["Проверка только для описей НТД"]

    doc_table.rename(columns={doc_table.columns[1]: "Номер"}, inplace=True)
    doc_table = doc_table.merge(df[df["contains_flag"]], on="Номер")

    table = prep_dataset_doc_10(table)

    table["Флаг"] = 0
    doc_table["Раздел"] = doc_table["Раздел"].apply(lambda x: x.lower().split())
    doc_table["Название"] = doc_table["Вид документа"].apply(
        lambda x: x.lower().split()
    )

    comments = []

    # Используем ThreadPoolExecutor для параллельной обработки строк
    with ThreadPoolExecutor(4) as executor:
        futures = []
        for i in range(len(table["Заголовок дела"])):
            futures.append(
                executor.submit(process_row_10, i, table, doc_table, comments)
            )

        # Ожидаем завершения всех задач
        for future in as_completed(futures):
            future.result()  # Чтобы ловить возможные исключения, если они были в процессе выполнения

    # print(f"Время выполнения поиска: {int((time.time() - start_time1) // 60)} минут {(time.time() - start_time1) % 60:.2f} секунд")

    all_matched = (table["Флаг"] == 1).all()
    end_comment = []
    if not all_matched:
        comments_str = ", ".join(map(str, comments))
        end_comment = [
            f"Запись(и) № {comments_str} не соответствует(ют) перечню из справочника СП6"
        ]

    return all_matched, end_comment


@timer
def check_21(text):
    ending_texts = ["в данную опись внесено", "в данный раздел описи"]

    end_page_1 = find_number_end_page(text)
    end_page_2 = find_number_page(text, ending_texts[0])
    end_page_3 = find_number_page(text, ending_texts[1])

    if end_page_1 is not None:
        end_page = end_page_1
    elif end_page_2 is not None:
        end_page = end_page_2
    elif end_page_3 is not None:
        end_page = end_page_3
    else:
        return False

    text_ends = {text[end_page]}

    pattern_1 = r'[^\w]согласовано[^\w]'
    pattern_2 = r'[^\w]эк[^\w]'
    other_pages = {}

    if len(text) - 1 > end_page:
        text_ends = {text[end_page], text[end_page + 1]}

    for text_elem in text_ends:
        result_search_1 = re.search(pattern_1, text_elem, flags=re.IGNORECASE)
        result_search_2 = re.search(pattern_2, text_elem, flags=re.IGNORECASE)

        if (result_search_1 is not None) and (result_search_2 is not None):
            result_search_1_end = result_search_1.end()
            result_search_2_start = result_search_2.start()

            if result_search_1_end < result_search_2_start:
                count_words = len(text_elem[result_search_1_end:result_search_2_start].split())
                if count_words > 20:
                    return False
                return True

        elif result_search_1 is not None:
            other_pages['search_1'] = result_search_1.end()

        elif result_search_2 is not None:
            other_pages['search_2'] = result_search_2.start()

    if (not other_pages) or (len(other_pages) != 2):
        return False

    other_pages = list(other_pages.keys())
    if (other_pages[0] == 'search_1') and (other_pages[1] == 'search_2'):
        return True


@timer
def check_22(text, table):
    try:
        doc_ldate, doc_rdate = get_date_inventory(text)
    except Exception as err:
        return "Неудалось", "Неудалось получить диапазон дат описи: {err}"
    if doc_ldate == doc_rdate:
        doc_rdate = doc_ldate + relativedelta(years=1) 
    else:
        doc_rdate = add_one_month(doc_rdate) 
    
    bad_days = []
    bad_years = []
    for num, header, t in zip(table.iloc[:, 0], table.iloc[:, 2], table.iloc[:, 3]):
        if "," in t:
            bad_years.append(num)
        date_h = parse_header(header, morph)
        date_edge_source = str2date(t, morph)
        if date_h == None or date_edge_source == None:
            continue
        if not isinstance(date_edge_source, tuple):
            if date_edge_source.day == 1 and date_edge_source.month == 1:
                edge1 = date_edge_source
                edge2 = date_edge_source + relativedelta(years = 1) - relativedelta(days=1)
            else:
                edge1 = date_edge_source
                edge2 = date_edge_source
        else:
            if date_edge_source[1] == None: #если не распознали дату dd.mm.year-dd.mm.year
                bad_days.append(num)
            else:
                edge1 = date_edge_source[0]
                edge2 = date_edge_source[1]
                
        if isinstance(date_h, tuple):
            d1 = date_h[0]
            d2 = date_h[1]
        else:
            d1 = d2 = date_h

        if not d1.month == edge1.month and d2.month == edge2.month and d1.year == d2.year == edge1.year == edge2.year:
    
            bad_days.append(num)
        # if not(edge1 <= d1 <= edge2 and edge1 <= d2 <= edge2):
        #     bad_days.append(num)
        #вне интервала даты документа
        if not (doc_ldate.year <= edge1.year <= doc_rdate.year and doc_ldate.year <= edge2.year <= doc_rdate.year):
            # print(num, doc_ldate, edge1,edge2, doc_rdate)
            if num not in bad_days:
                bad_days.append(num)
        #проверка для 24
        try:
            if not(is_valid_data_header(header, morph)):
                if num is not bad_years:
                    bad_years.append(num)
                    
            if is_last_date(t):#01.01.****-31.12.****
                if num is not bad_years:
                    bad_years.append(num)
        except Exception as err:
            print(err)
            
        # if not(is_valid_data_header(header, morph)):
        #         bad_years.append(num)
        # if is_last_date(t):#01.01.****-31.12.****
        #     if num is not bad_years:
        #         bad_years.append(num)
        #проверка крайних дат
    ch22 = True if len(bad_days) == 0 else False
    ch24 = True if len(bad_years) == 0 else False
    if ch22:
        com1 = ""
    else:
        com1 = "Некорретные даты в:" + " ,".join(bad_days)

    if ch24:
        com2 = ""
    else:
        com2 = "Некорретные даты в:" + " ,".join(bad_years)
        
    return ch22, ch24, com1, com2

@timer
def check_23(table):
    """Проверка наличия номера в графе «Заголовок ...» при типах документа «приказы», «распоряжения», «постановления», «протоколы», «решения»"""

    def check(row):
        if bool(
            re.search(
                "((приказ)|(распоряжен)|(постановлен)|(протокол)|(решен))[а-яА-Я ,]+№\s*\d+",
                row.iloc[2].lower(),
            )
        ):
            return True
        else:
            if row.iloc[-1].strip() == "":
                return False
            return True

    tab = table[
        table.iloc[:, 2].apply(
            lambda x: bool(
                re.search(
                    "^((приказ)|(распоряжен)|(постановлен)|(протокол)|(решен))",
                    x.lower(),
                )
            )
        )
    ].copy()
    if len(tab) == 0:
        return (
            True,
            'Заголовки не содержат ключевых слов: "приказ", "распоряжение", "постановление", "протокол", "решение".',
        )
    tab["result"] = tab.apply(check, axis=1)
    res_t = tab[tab["result"] == False]
    res = len(res_t)
    return res == 0, (
        ""
        if not res
        else "Некорректные заголовки №: " + ", ".join(res_t.iloc[:, 0].tolist())
    )


@timer
def check_25(dict_text):
    """
    Проверка наличия списка сокращений в описи дел
    """

    def find_abbreviations(text):
        # Поиск сокращений, состоящих из строчных букв (1-3 буквы) с точкой
        short_abbrs = re.findall(r"\b[а-яёa-z]{1,3}\.\s?", text)
        # Поиск аббревиатур, состоящих из двух и более прописных букв
        full_abbrs = re.findall(r"\b[A-ZА-ЯЁ]{2,4}\b", text)
        return short_abbrs, full_abbrs

    flag = False
    abbr_text = set()
    abbr_list = set()

    for i in range(1, len(dict_text)):
        # if 'УТВЕРЖДАЮ' in dict_text[i]:
        #     break
        if (
            "СПИСОК СОКРАЩЕННЫХ СЛОВ" in dict_text[i]
            or "СПИСОК СОКРАЩЕНИЙ" in dict_text[i]
            or "Список сокращений" in dict_text[i]
            or "СПИСОК АББРЕВИАТУР И СОКРАЩЕНИЙ" in dict_text[i]
        ):
            flag = True
            short_abbr, full_abbr = find_abbreviations(dict_text[i])
            abbr_list.update([abbr.rstrip() for abbr in short_abbr])
            abbr_list.update([abbr.rstrip() for abbr in full_abbr])

        short_abbr, full_abbr = find_abbreviations(dict_text[i])
        abbr_text.update([abbr.rstrip() for abbr in short_abbr])
        abbr_text.update([abbr.rstrip() for abbr in full_abbr])

    # if flag and abbr_text == abbr_list:
    #      return True
    # else:
    #     return False

    if flag == True and len(abbr_text) != 0 and len(abbr_list) != 0:
        return True
    else:
        return False


@timer
def check_26(dict_text):
    """
    Проверка наличия подписанного «предисловия» или «дополнения к предисловию» в описи.
    Нужна проверка блока с подпью специалиста.
    """

    end_pred = find_number_page(dict_text, "утверждаю")
    pattern = r"\b(предислови(е|ю)|к предисловию|к исторической справке|историческая справка|дополнение)\b"
    pages_start = []
    for num in range(end_pred):
        if re.search(pattern, dict_text[num].lower()):
            pages_start += [num]
    if len(pages_start) == 0:
        return False, "Предисловие не найдено"
    start_pred = pages_start[0]
    has_history_fond_start = False
    has_history_fond = False

    for i in range(start_pred, end_pred):
        if re.findall("история {1,20}фондообра", dict_text[i].lower()):
            has_history_fond_start = True
        if re.findall("история {1,20}фонда", dict_text[i].lower()):
            has_history_fond = True

    if has_history_fond_start and has_history_fond > 0:
        return True, ""
    else:
        return False, "История фонда и фондообразователя не найдены"


@timer
def check_28(text):

    title = text[0]
    pattern = r"\b(предислови(е|ю)|к предисловию|к исторической справке|историческая справка|дополнение)\b"
    matches1 = re.findall(pattern, text[1].lower(), re.IGNORECASE)
    matches2 = re.findall(pattern, text[2].lower(), re.IGNORECASE)
    if matches1 == [] and matches2 == []:
        return True, 'Не применимо' 
    if matches1 != []:
        num = 1
    else:
        num = 2
    end_pred = find_number_page(text, 'утверждаю')
    list_keys = list(range(num,end_pred))
    pred = ' '.join(text[key] for key in list_keys if key in text)
    fio = list(names_extractor(pred))[-1]
    pred = pred[:fio.stop]

    # Поиск дат на титуле
    proper_dates = extract_proper_dates(title)
    if proper_dates == False:
        return False, ''
    
    if 'история фонда' in text[num].lower() or 'упорядочение документов' in text[num].lower():
        date_ = datetime.now().year
        proper_dates.append(date_)

    # Поиск всех дат в предисловии
    year_pattern = r"\b(19[5-9][0-9]|20[0-2][0-9]|2100)\b"
    matches3 = re.findall(year_pattern, pred)
    years_pred = set(int(year) for year in matches3)
    all_elements_in_set = years_pred.issubset(set(proper_dates))
    return all_elements_in_set, ''

@timer
def check_27(text, fpath):
    text, full_text, page_blocks = read_text_documents(fpath)
    start_page = 0
    for substr in ["предислови", "дополнение", "история фондообразователя"]:
        sp = find_number_page(text, substr)
        if sp is None:
            continue
        if sp > start_page:
            start_page = sp

    end_page = find_number_page(text, "утверждаю")
    history_page = find_number_page(text, "история {1,20}фонда", -1)
    pages = []
    if history_page is not None:
        if start_page < history_page < end_page:
            start_page = history_page
    if start_page is None or end_page is None:
        return False, "Предисловие не найдено"
    pdf = pdfplumber.open(fpath)

    if not start_page is None and not end_page is None:
        if end_page > start_page:
            for i in range(start_page, end_page):
                pages.append(pdf.pages[i])
        else:
            pdf.close()
    page_to_candidate_lines = {}
    for page in pages:
        lines = page.extract_text_lines()
        lines_with_candidates = []
        for line in lines:
            doc = Doc(line["text"])
            doc.segment(segmenter)
            doc.tag_ner(ner_tagger)
            # doc.tag_morph(morph_tagger)
            for span in doc.spans:
                span.normalize(morph_vocab)
            for span in doc.spans:
                if span.type == PER:
                    span.extract_fact(names_extractor)
                    lines_with_candidates.append(line)
        page_to_candidate_lines[page] = lines_with_candidates

    found_specialists = []
    for page in page_to_candidate_lines:
        for candidate in page_to_candidate_lines[page]:
            h = candidate["bottom"] - candidate["top"]
            y0 = candidate["top"] - h * 0.3
            y1 = candidate["bottom"] + h * 0.3
            width = page.width
            page_crop = page.crop([width / 2, y0, width, y1])
            if len(page_crop.extract_words()) < 10:
                found_specialists.append({"page": page.page_number, "line": candidate})
    if len(found_specialists) == 0:
        return False, "Должностные лица к предисловию не найдены"
    max_page = np.max([x["page"] for x in found_specialists])
    found_specialists = list(filter(lambda x: x["page"] == max_page, found_specialists))

    search_specialists = []
    if len(found_specialists) > 2:
        for spec in found_specialists:
            search_specialists.append(
                {"h": page.height - spec["line"]["top"], "line": spec["line"]}
            )
        search_specialists = sorted(search_specialists, key=lambda x: x["h"])[:2]
    else:
        search_specialists = found_specialists
    pdf.close()
    if len(search_specialists) > 0:
        return True, ""
    else:
        return False, "Должностные лица к предисловию не найдены"


@timer
def check_0(path):
    """
    Проверка таблицы
    """

    text, full_text, page_blocks = read_text_documents(path)
    number_begin = get_begin_page(text)
    if number_begin < 0:
        return False, "Неудалось найти страницу начала таблицы в документе"

    number_end1 = find_number_page(text, END_PAGE_PATTERN, -1)
    number_end2 = find_number_page(text, END_PAGE_PATTERN2, -1)
    if number_end1 is not None:
        number_end = max(number_end1, number_end2)
        if find_number_page(text, 'пояс[-\\w\\d\\D №()А-я]{0,25}запи',-1)  == number_end2:
            number_end = number_end1
    else:
        number_end = number_end2
    pdf_file = pdfplumber.open(path)

    table_settings = {
        # "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
    }

    count_board_line = count_board_line_in_page(pdf_file.pages[number_begin])
    values = {}
    filter_vlines, has_board, y_begin, y_end = get_vertical_lines(
        pdf_file.pages[number_begin]
    )

    has_header_end = False
    if number_end > number_begin:
        if re.findall("индекс", text[number_end].lower()) and re.findall(
            "заголов", text[number_end].lower()
        ):
            has_header_end = True

    if number_end - number_begin > 100:
        comment_0 = ""
        if not has_board:
            comment_0 = "".join((comment_0, "Нет боковых граней;\n"))
        if has_header_end:
            comment_0 = "".join((comment_0, "Дублирование заголовка таблицы;\n"))
        if count_board_line > 1:
            comment_0 = "".join(
                (comment_0, "Две шапки номеров столбцов на странице начала таблицы;\n")
            )

        if has_board == True and has_header_end == False and count_board_line == 1:
            return True, ""
        else:
            return False, ""

    for number in range(number_begin + 1, number_end + 1):
        page = pdf_file.pages[number]
        crop = page.crop((0, 0, page.width, page.height*0.17))

        for x in crop.extract_text_lines():

            s = re.findall("\d", x["text"])
            w = re.findall("[А-я]", x["text"])
            if len(s) > 4 and len(w) < 3:
                board = np.array(s, dtype=np.int32)
                values[number] = board
    page_without_header = []
    if number_begin + 1 < number_end:
        for i in range(number_begin + 1, number_end + 1):
            if not i in values:
                page_without_header.append(i)

    comment_0 = ""
    if not has_board:
        comment_0 = "".join((comment_0, "Нет боковых граней;\n"))
    if len(page_without_header) > 0:
        comment_0 = "".join((comment_0, "Не найден заголок-разделитель на страницах:"))
        a = "; ".join([str(x) for x in page_without_header])
        comment_0 = "".join((comment_0, a, "\n"))
    if has_header_end:
        comment_0 = "".join((comment_0, "Дублирование заголовка таблицы;\n"))
    if count_board_line > 1:
        comment_0 = "".join(
            (comment_0, "Две шапки номеров столбцов на странице начала таблицы;\n")
        )

    if (
        has_board == True
        and len(page_without_header) == 0
        and has_header_end == False
        and count_board_line == 1
    ):
        return True, comment_0
    else:
        return False, comment_0
