import numpy as np
import pdfplumber
import pdfplumber.page
# from text_utils import find_number_page, get_begin_table, read_text_documents
from src.app.checker.utils.text_utils import find_number_page, get_begin_table, read_text_documents
from src.app.checker.const import END_PAGE_PATTERN, END_PAGE_PATTERN2
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from src.app.checker.utils.file_utils import timer
from concurrent.futures import ProcessPoolExecutor, as_completed
import PyPDF2
import os
from pathlib import Path
import pymorphy2

MORPH = pymorphy2.MorphAnalyzer()
STOP_WORDS = stopwords.words('russian')


def join_strings(x):
    return ''.join(x)

def can_convert_to_float_extended(s):
    try:
        f = float(s)
        return True
    except ValueError:
        return s.lower() in {'nan', 'inf', '-inf', 'infinity', '-infinity'}


def is_destination_number(s:str):
    """
        Проверка соотствует строка номеру пункта из таблицы описи
    """
    return can_convert_to_float_extended(s)


def get_vertical_lines(page):
    table_settings_b = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }

    tables = page.find_tables(table_settings_b)
    table, y_begin, y_end = get_begin_table(tables)
    crop=page.crop((0, y_begin-3, page.width, y_end+3))

    # v_lines = np.sort(np.unique([x['x0'] for x in crop.vertical_edges if x['x0']>5 and x['x0']<crop.width - 2 and abs(x['y0'] - x['y1']) > 3]))
    v_lines = []
    for x in crop.vertical_edges:
        if 'x0' in x and 'y0'in x and 'y1' in x:
            if x['x0']>5 and x['x0']<crop.width - 2 and abs(x['y0'] - x['y1']) > 3:
                v_lines.append(x['x0'])
    v_lines = np.unique(v_lines)
    #-----
    v_out = []
    for x in crop.edges:
        v_out.append(x['x0'])
        v_out.append(x['x1'])
    v_out = np.sort(v_out)
    v_out = v_out[v_out > 10]
    filter_v_out = []
    for i in range(0,len(v_out)):
        if abs(v_out[i-1] - v_out[i]) > 10:
            filter_v_out.append(v_out[i])
            
    filter_vlines = []
    for i in range(0,len(v_lines)):
        if abs(v_lines[i-1] - v_lines[i]) > 10:
            filter_vlines.append(v_lines[i])
    left_out  = np.min(filter_v_out)
    right_out = np.max(filter_v_out)
    has_board = True
    if np.min(filter_vlines) - left_out > 2  and right_out - np.max(filter_vlines) > 2:
        has_board = False
        filter_vlines.append(left_out)
        filter_vlines.append(right_out)
    return filter_vlines, has_board, y_begin, y_end

def get_base_row(frame):
    """
        Получить опорную строку таблицы с цифрами [1,2,3...n]
        return:
            number_row: int, номер базовой строки
            base_row: List[str], базовая строка таблица
            max_len_row:[int], максимальная длина строки таблицы
    """
    max_len_row = 0
    number_row = 0
    base_row = frame.iloc[:,0].tolist()
    for k,row in frame.iterrows():
        if len(row) > max_len_row:
            max_len_row = len(row)
        values = []
        for e in row.tolist():
            try:
                v = int(e)
                values.append(v)
            except:
                pass
        values = np.array(values)

        if len(values) > 3 and len(values)-2<=np.diff(values).sum() <= len(values)+2:
            base_row = row.tolist()
            number_row = k
    return number_row, base_row, max_len_row
    
def concate_table_by_rows(frame, base_row):
    """
        Объединить строки в ячейках по строкам
    """
    last_n = 0
    index = 0
    df_concat = pd.DataFrame()

    for k,el in enumerate(base_row):

        if el is None:
            pass
        else:
            merge = pd.DataFrame(frame.iloc[:,last_n:k].fillna('').apply(lambda row: ' '.join(row), axis=1))
            merge.columns = [index]
            index+=1
            df_concat = pd.concat((df_concat,merge),axis=1)
            last_n = k
    df_concat = pd.concat((df_concat,frame.iloc[:,k:]),axis=1)
    return df_concat.iloc[:,1:]


def collect_table_lines(df, number_base):
        """
            Collecting the summary table by rows and merging rows
        """
        df.iloc[:,0] = df.iloc[:,0].replace(r'\s\s+', '', regex=True)
        df.fillna("", inplace = True)
        if number_base > 1:
            header = df.iloc[:number_base,:].fillna('')
        else:
            header = pd.DataFrame()
        new_header = {}
        for k,(s, series) in enumerate(header.items()):
            new=series.replace('\n','').str.cat(sep=' ').strip()
            new_header[k] = new
        header = pd.DataFrame(data = new_header.values(),index = new_header.keys()).T
        last_line_number = ''
        collect_lines = []
        for k,row in df.copy().iterrows():
            if set(row) == {''}:
                continue
            if k <= number_base:
                pass
            else:
                cur_number = row.values[0]
                if is_destination_number(cur_number):
                    vals = row.values
                    if vals[0] != '':
                        last_line_number = cur_number
                    collect_lines.append(list(row))
                if cur_number == '':
                    #большие буквы в строке
                    if join_strings(row).isupper():
                        collect_lines.append(['','',join_strings(row),'','',''])
                    else:
                        row[1] = last_line_number
                        collect_lines.append(list(row))

        new_df = pd.DataFrame(collect_lines)
        new_df.replace(np.nan,'',inplace = True)

        res = pd.concat((
                        # new_df.iloc[:number_base-1,:],
                        header,
                        # new_df.iloc[:number_base-1,:].groupby(new_df.columns[0]).agg(' '.join).reset_index(),
                        new_df.groupby(new_df.columns[0]).agg(' '.join).reset_index())) #конкантенация строк по столбцу "№ п/п"
        # res.reset_index(inplace= True)
        return res, collect_lines, new_df, header

def get_table_begin_page(page_begin):
    v_lines, has_boarder, y_begin, y_end = get_vertical_lines(page_begin)#y_begin - начало y таблицы на странице
    y_top = page_begin.height
    for x in page_begin.extract_text_lines():
        if re.findall(f"{END_PAGE_PATTERN}|{END_PAGE_PATTERN2}", x['text']):
            y_top = x['top']
    page_crop = page_begin.crop([ 0, y_begin, page_begin.width, y_top])
    page_crop = page_crop.filter(lambda x: x['object_type'] == 'char')
    table_settings_b = { "vertical_strategy": "lines",
                         "horizontal_strategy": "text",
                         "explicit_vertical_lines": v_lines,
                         "intersection_tolerance": 25}
    result  = pd.DataFrame(page_crop.find_tables(table_settings_b)[0].extract())
    num_base, base, maxlen =get_base_row(result)
    temp = concate_table_by_rows(result,base)
    clear, lines, r,t = collect_table_lines(temp,num_base)
    return clear, has_boarder, v_lines


def get_table_end_page(v_lines, page_end):
    table_settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "text",
    "explicit_vertical_lines": v_lines,
    "intersection_tolerance": 25
}
    #кропнуть по низу
    y_top = page_end.height
    for t in page_end.extract_text_lines():
        if re.search('[в|В][-\w №()А-я]{1,100}опис[-\w\d\D №()А-я]{1,100}внесен',t['text']):
            y_top = t['top'] - 5
    p_crop = page_end.crop((0, 0, page_end.width, y_top))
    p_crop = p_crop.filter(lambda x: x['object_type'] == 'char')
    t_ext = p_crop.extract_table(table_settings)
    table_row = pd.DataFrame(t_ext)

    # table_row.replace('', np.nan, inplace=True)
    # table_row.dropna(how='all', axis  = 1, inplace=True)
    # table_row.replace('None', np.nan, inplace=True)
    # table_end = table_row.replace(np.nan, '')
    # table_row  = drop_table_lines(table_row)
    num_base, base, maxlen =get_base_row(table_row)
    temp = concate_table_by_rows(table_row,base)
    clear, lines, r,t = collect_table_lines(temp,num_base)
    return clear


def get_table_page(v_lines, page):
    def string_of_spaces(string):
        return len(string) == len(np.array(re.findall("\s", string)) == " ")
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "text",
        "explicit_vertical_lines": v_lines,
        "intersection_tolerance": 25
    }
    page = page.filter((lambda x: x['object_type'] == 'char'))
    table_end = pd.DataFrame(page.extract_table(table_settings))
    table_end.replace('', np.nan, inplace=True)
    table_end.dropna(how='all', inplace=True)
    table_end.replace('None', np.nan, inplace=True)
    table_end.replace(np.nan, '', inplace=True)

    last_line_number = ''
    collect_lines = []
    for k,row in table_end.copy().iterrows():
        cur_number = row[0]
        n = len(row)
        if list(row) == [str(x) for x in range(1, n+1)]:
            continue
        items = list(row.values)
        if k == 1 and sum([string_of_spaces(x) for x in items]) > 3:
            #Строка из диапазона дат, удалить
            flag_years = False
            for x in items:
                if (re.findall("\d{4}-\d{4}", x) and len(x) == 9) or (re.findall("\d{4}", x) and len(x) == 4):
                    flag_years = True
            if flag_years:
                continue
        if is_destination_number(cur_number):
            last_line_number = cur_number
            collect_lines.append(list(row))
        if cur_number == '' or cur_number == ' ':
            # большие буквы в строке
            if join_strings(row).isupper():
                collect_lines.append([last_line_number,'',join_strings(row),'','',''])
            else:
                row[0] = last_line_number
                collect_lines.append(list(row))      
    # return collect_lines
    for m,line in enumerate(collect_lines):
        if line[0] == "":
            if m > 0:
                index = collect_lines[m-1][0]
            else:
                f = m + 1
                #поиск индекса для замена в строке_____________
                while f < len(collect_lines):
                    if collect_lines[f] != "" and re.findall("\d", collect_lines[f][0]):
                        index = collect_lines[f][0]
                        break
                    else:
                        f+=1
                #_________________________
            new_line = line.copy()
            new_line[0] = index
            collect_lines[m] = new_line
            
    # for l in collect_lines:
        # print(l)
    new_df = pd.DataFrame(collect_lines)
    new_df.replace(np.nan,'', inplace = True)
    new_df.index = new_df.index.astype(int)
    table = new_df.groupby(new_df.columns[0]).agg(' '.join) #конкантенация строк по столбцу "№ п/п"
    table.index = table.index.astype(int)
    table = table.sort_index(ascending=True).reset_index()
    return table


def check_table_header(table):
    """
    table 
    Returns:
    ----------
        bad_columns: list - cписок неверных колонок
    """
    bad_columns = []
    header = table.iloc[0].fillna("")
    shape = table.shape
    if shape[1] == 6:
        columns = ['№', "индекс", "заголов", "крайн", "листов" , "меча"]
        for i,c in enumerate(header):
            if not columns[i] in re.sub('[-\s]',"", c.lower()):
                bad_columns.append(c)
    elif shape[1] == 7:
        columns = ['№', "индекс", "заголов", "крайн", "хранен", "листов" ,  "меча"]
        for i,c in enumerate(header):
            if not columns[i] in re.sub('[-\s]',"", c.lower()):
                bad_columns.append(c)
    else:
        bad_columns = header.tolist()
    return bad_columns


def get_begin_page(text):
    page_begin = -1
    for k, v in text.items():
        t = v.lower()
        c = 0
        for p in ['опись', 'заголов', 'утверждаю']:
            if p in t:
                c+=1
        if c == 3:
            page_begin = k
        if "О П И С Ь" in v:
            page_begin = k
    return page_begin


@timer
def get_table_documents(path):
    text, full_text, page_blocks = read_text_documents(path)
    pdf_res = pdfplumber.open(path)

    number_begin = get_begin_page(text)
    if number_begin < 0:
        return None, 'Неудалось найти страницу начала таблицы в документе'
    number_end1  = find_number_page(text, END_PAGE_PATTERN, -1)
    number_end2  = find_number_page(text, END_PAGE_PATTERN2,-1)
    if number_end1 is not None:
        number_end = max(number_end1, number_end2)
        if find_number_page(text, 'пояс[-\\w\\d\\D №()А-я]{0,25}запи',-1)  == number_end2:
            number_end = number_end1
    else:
        number_end = number_end2
    page_begin = pdf_res.pages[number_begin]
    page_end = pdf_res.pages[number_end]
    first_page_table, has_boarder, v_lines = get_table_begin_page(page_begin)

    bad_columns = check_table_header(first_page_table)
    last_page_table = get_table_end_page(v_lines, page_end)
    pages = pd.DataFrame()
    
    if number_end - number_begin < 100:
        for i in range(number_begin + 1, number_end):
            other = get_table_page(v_lines, pdf_res.pages[i])
            pages = pd.concat([pages, other ])
    else: #несколько потоков
        f_paths = split_pdf(path, Path(path).stem, 80, number_begin + 1, number_end)
        pages = merge_table_from_paths(f_paths, v_lines)
        # df_pages = {}
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     futures = {executor.submit(get_table_page, v_lines, pdf_res.pages[i]): i for i in range(number_begin + 1, number_end)}
        #     for future in as_completed(futures):
        #         input_arg = futures[future]
        #         try:
        #             df_pages[input_arg] = future.result()
        #         except Exception as e:
        #             print(f"Обработка страницы {input_arg} не удалась")
        #     pages = pd.DataFrame()
        #     for i in sorted(df_pages, key=lambda x: x):
        #         pages = pd.concat([pages, other ])

    if page_begin == page_end: #если таблица на одной странице документа
        pages = pd.concat([first_page_table, pages]).reset_index(drop=True).fillna('')
        return pages, bad_columns
    else:
        pages = pd.concat([first_page_table, pages, last_page_table]).reset_index(drop=True).fillna('')
    pages.iloc[0] = pages.iloc[0].apply(lambda x: x.replace('\n', ' ').replace('- ', ''))
    pages.columns = pages.iloc[0]

    for i, x in enumerate(pages.iloc[:,0]):
        if len(re.findall('\d+',x)) > 0:
            start_row = i
            break
        else:
            start_row = 0
    pages.drop(pages.index[0:start_row], inplace=True)
    pages = pages.reset_index(drop=True).fillna('').astype(str)
    # const_cols = ['№', 'Индекс дела', 'Заголовок дела', 'Крайние даты', 'Кол-во листов', 'Примечание']
    # cols = list(pages.columns)
    # indexes = [i for i, x in enumerate(cols) if x == '']
    # for i in indexes:
    #     cols[i] = const_cols[i]
    # pages.columns = cols
    # cols.remove('Заголовок дела')
    # pages['Заголовок дела'] = np.where(pages[pages.columns[0]] == '', pages.sum(axis=1), pages['Заголовок дела'])
    # for col in cols:
    #     pages[col] = np.where(pages[pages.columns[0]] == '', '', pages[col])
    return pages, bad_columns

def drop_table_lines(table):
    """
       Удалить лишние строки из вытащенной таблицы. Удаляем [['1','2','3', ...],  [['','','1.Организационно – распорядительная деятельность', '', ..] и т.д.
    """
    used_lines = []
        
    def string_of_spaces(string):
        return len(string) == len(np.array(re.findall("\s", string)) == " ")
    for k, line in table.iterrows():
        items = list(line.values)
        if sum([string_of_spaces(x) for x in items]) > 4:
            continue
        if sum(line.values == "") >= 4:
            continue    
        if all(line.values[:5] == ['1','2','3','4','5']):
            continue
        try:
            if all(np.array([int(x) for x in line.values[-4:]]) == np.arange(1,table.shape[1]+1)[-4:]):
                continue
        except:
            pass
        used_lines.append(line.values)
    res = pd.DataFrame(used_lines)
    if not is_destination_number(res.iloc[0,0]):#первая строка не заголовок таблицы
        res = res.iloc[1:,:]
    return res

def get_tocken(text, norm_form=False):
    '''Вспомогательная функция для очистки от стоп слов и приведения слов к нормальной форме'''
    text = re.sub('[/\\\]', ' ', text)
    tokens = word_tokenize(text)
    lemma_transform = WordNetLemmatizer()
    tokens = [lemma_transform.lemmatize(token) for token in tokens]
    tokens = [x for x in tokens if x.isalnum()]
    sen = ' '.join([MORPH.parse(token)[0].normal_form for token in tokens if token not in STOP_WORDS]) if norm_form\
                else ' '.join([token for token in tokens if token not in STOP_WORDS])
    return sen


def count_board_line_in_page(page:pdfplumber.page)-> int:
    """
        Количество шапок начала таблицы на странице
    """
    def f_check(x):
        if int(x[0]) == 1:
            if int(x[1]) == 2:
                return True
        elif int(x[0]) == 2:
            if int(x[1]) == 3:
                return True
            else:
                return False
        else:
            return False
            
    values = []
    for table in page.extract_tables():
        for x_line in table:
            if len(x_line) > 3:
                xs = [str(e) for e in x_line if e is not None]
                merge="".join(xs)
                s = re.findall("\d", merge)
                w = re.findall("\D", merge)
                if len(s) >= 4 and len(w) < 3 and f_check(xs):
                    values.append(x_line)
    return len(values)


def split_pdf(input_pdf_path, output_prefix, pages_per_part, start, end):
    with open(input_pdf_path, 'rb') as input_pdf_file:
        reader = PyPDF2.PdfReader(input_pdf_file)
        total_pages = end
        file_paths = []
        for part_num in range(start, end, pages_per_part):
            
            writer = PyPDF2.PdfWriter()
            for page_num in range(part_num, min(part_num + pages_per_part, total_pages)):
                writer.add_page(reader.pages[page_num])
            
            output_pdf_path = os.getcwd() + "/src/app/data/" + f"{output_prefix}_part_{part_num // pages_per_part + 1}.pdf"
            with open(output_pdf_path, 'wb') as output_pdf_file:
                writer.write(output_pdf_file)
                file_paths.append(output_pdf_path)
            print(f"Сохранена часть: {output_pdf_path}")
    return file_paths


def collect_table_pages_from_file(path2file, v_lines):
    pdf_res = pdfplumber.open(path2file)
    pages = pd.DataFrame()
    for page in pdf_res.pages:
        other = get_table_page(v_lines, page)
        pages = pd.concat([pages, other])
    return pages

def merge_table_from_paths(f_paths,v_lines):
    df_pages = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(collect_table_pages_from_file, p, v_lines):p for p in f_paths}
        for future in as_completed(futures):
            input_arg = futures[future]
            try:
                df_pages[input_arg] = future.result()
            except Exception as e:
                print(f"Обработка страницы {input_arg} не удалась")
    df = pd.DataFrame()
    for k in sorted(df_pages, key=lambda x: x):
        os.remove(k) 
        df = pd.concat((df,df_pages[k]))
    return df