# изменил функции get_pkl_fname, append
import os
import sys
from pathlib import Path
from pandas import DataFrame
import pickle as pkl
from typing import Union
import shutil
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.app.checker.utils.text_utils import prep_sp_lists, prep_sp_lists_fast, prep_sp6_tables
import numpy as np
import pandas as pd
import re
import pickle
from typing import List
import logging
from src.app.checker.checker import names_checks

def path2directory():
    platform = sys.platform
    if platform == 'win32':
        p = os.getcwd() + "\\src\\app\\data\\work_lists\\"
    else:
        # p = os.getcwd() + "/src/app/sps/"
        p = os.getcwd() + "/src/app/data/work_lists/"
    return p

def path2backup():
    platform = sys.platform
    if platform == 'win32':
        p = os.getcwd() + "\\src\\app\\backup\\"
    else:
        p = os.getcwd() + "/src/app/backup/"
    if not Path(p).exists():
        Path(p).mkdir()
    return p

def path2save():
    platform = sys.platform
    if platform == 'win32':
        p = os.getcwd() + "\\src\\app\\save\\"
    else:
        p = os.getcwd() + "/src/app/save/"
    if not Path(p).exists():
        Path(p).mkdir()
    return p
#backup всех файлов при инициализации

def get_pkl_fname(number=1, sub_number=0):
    if sub_number == 0:
        if number == 4:
            f_name = f"sp{number}_proc.pkl"
        if number == 5:
            f_name = [f"sp{number}.pkl", f"sp{number}_list.pkl"]
        if number == 7:
            f_name = "sps7_14_merge.pkl"
        if number == 8:
            f_name = "sps15_21_merge.pkl"
        elif number not in {4, 5, 7, 8}:
            f_name = f"sp{number}.pkl"
    else:
        if number == 6:
            f_name = [f"sp{number}_{sub_number}.pkl", f"sp6_list.pkl"]
        else:
            f_name = f"sp{number}_{sub_number}.pkl"
    
    return f_name


def correct_frame_list(frame: pd.DataFrame, frame_list: pd.DataFrame):
    """
        Внести изменения в sp_{5,6}_list.pkl
    """
    comments = []
    for i in range(frame.shape[0]):
        try:
            frame_list.loc[frame_list.shape[0]] = ["".join(frame.iloc[i, 1].split()[1:]),
                                                                    str(frame_list.shape[0])] +\
                        [frame.iloc[i, 1].split()[0].upper()] + [""] * (frame_list.shape[1] - 3)
        except Exception as err:
            comments.append(err)
    return frame_list, comments


class SpManager:
    def __init__(self, path2sp: Union[str, Path], dbmanager):
        self.root_path = Path(path2sp)
        self.dbase = dbmanager
        self.pkl_names = {
                    1:["sp1.pkl"],
                    2:["sp2.pkl"],
                    3:["sp3.pkl"],
                    4:["sp4.pkl"],
                    5:["sp5.pkl", "sp5_prep.pkl"],
                    # 6:[],
                    7:["sp7.pkl"],
                    8:["sp8.pkl"],
                    9:["sp9.pkl"],
                    10:["sp10.pkl"],
                    11:["sp11.pkl"],
                    12:["sp12.pkl"],
                    13:["sp13.pkl"],
                    14:["sp14.pkl"],
                    15:["sp15_1.pkl","sp15_2.pkl"],
                    16:["sp16_1.pkl","sp16_2.pkl"],
                    17:["sp17_1.pkl","sp17_2.pkl"],
                    18:["sp18.pkl"],
                    19:["sp19_1.pkl","sp19_2.pkl"],
                    20:["sp20.pkl"],
                    21:["sp21.pkl"]}
        self.backup_files = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler('sp_manager_log.txt')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info("Sp manager create")
    
    def path2sp(self, indx: int)->List[Path]:
        if indx in self.pkl_names:
            return [self.root_path / p for p in self.pkl_names[indx]]
        else:
            return [None]
    @property
    def files_in_root(self)->List[str]:
        return [x.name for x in self.root_path.glob("*")]
        
    def is_exist(self, indx: int)->bool:
        if not indx in self.pkl_names:
            return False
        file_names = self.pkl_names[indx]
        for file in file_names:
            if not file in self.files_in_root():
                return False
        return True
        
    def backup_file(self, indx:int):
        names = self.pkl_names[indx]
        backups = []
        for name in names:
            new_name = (name.split('.')[0] + '.bck')
            shutil.move(self.root_path / name, self.root_path / new_name) 
            backups.append(new_name)
        self.backup_files[indx] = backups
        
    def recover_file(self, indx:int):
        for f in self.backup_files[indx]:
            shutil.copy(self.root_path / f, self.root_path / (f.split('.')[0] + '.pkl')) 
    
    def save_csv(self, indx: int, subindx:int = 0)->str:
        """"
            Сохранить в csv файл справочника
            return:
            fsave - path to csv
        """
        f_name = get_pkl_fname(indx, subindx)
        if isinstance(f_name, List):
            f_name = f_name[0]
        f = self.root_path / f_name
        self.logger.debug(f"Получить справочник {f_name}, indx= {indx}, subindx={subindx}")
        if not f.name in self.files_in_root:
            self.logger.debug(f"Нет файла {f_name} в директории {self.root_path}")
            raise AssertionError(f"Нет файла {f_name} в директории {self.root_path}")
        
        fsave = self.root_path / (f.name[:-4] + ".csv")
        with open(f,'rb') as handler:
            dataframe = pkl.load(handler)
            dataframe.to_csv(fsave, index=False, sep=";", encoding="cp1251")
        return fsave
    
    def save_pattern(self, indx:int, subindx:int = 0):
        """"
            Сохранить csv файл шаблон структуры справочника
            return:
            fsave - path to csv
        """
        f_name = get_pkl_fname(indx, subindx)
        if isinstance(f_name, List):
            f_name = f_name[0]
        f = self.root_path / f_name
        fsave = self.root_path / (f.name[:-4] + ".csv")
        self.logger.debug(f"Добавить новые строки в {f}, indx= {indx}, subindx={subindx}")
        if not f.name in self.files_in_root:
            raise AssertionError(f"Нет файла {f_name} в директории {self.root_path}")
        with open(f, 'rb') as handler:
            dataframe = pkl.load(handler)
            if indx == 2:
                df = dataframe.head(1)
            else:
                df = pd.DataFrame(columns=dataframe.columns)
            df.to_csv(fsave, index=False, sep=";", encoding="cp1251")
        return fsave

    def update_sp_list(self, frame: pd.DataFrame, indx: int = -1):  # обновить sp_list
        if indx in {5, 6}:
            f1, f2 = get_pkl_fname(indx)
            f1 = self.root_path / f1  # sp5,6
            f2 = self.root_path / f2  # sp5,6_list
            source_frame_1 = dataframe_from_pkl(f1)
            source_frame_2 = dataframe_from_pkl(f2)
            if is_equal(source_frame_1, frame):
                for i in range(frame.shape[0]):
                    source_frame_2.loc[source_frame_2.shape[0]] = ["".join(frame.iloc[i, 1].split()[1:]),
                                                                   str(source_frame_2.shape[0])] +\
                        [frame.iloc[i, 1].split()[0].upper()] + [""] * (source_frame_2.shape[1] - 3)
                dump_pkl(source_frame_2, f2)
            return True
        return False
    
    def summary(self):
        """
            Краткая статистика по справочникам в системе
        """
        lens = {}
        count_dup = {}
        for k in {1,2,3,4,7,8}:
            f1 = get_pkl_fname(k,0)
            frame_1 = pd.read_pickle(self.root_path/f1)
            lens[k] = int(len(frame_1))
            count_dup[k] = int(frame_1.duplicated().sum())
            if k in {7}:
                count_7 = frame_1.groupby("номер_справочника").count()
                info_7 = count_7.to_dict()['Заголовок дела']
            if k in {8}:
                count_8 = frame_1.groupby("номер_справочника").count()
                info_8 = count_8.to_dict()['Заголовок дела']
        
        summary = {"СП1":{"len":lens[1],"dup": count_dup[1]},
                "СП2":{"len":lens[2],"dup": count_dup[2]},
                "СП3":{"len":lens[3],"dup": count_dup[3]},
                "СП4":{"len":lens[4],"dup": count_dup[4]},
                "СП5":{"len": "Не реализовано","dup": "Не реализовано"},
                "СП6":{"len": "Не реализовано","dup": "Не реализовано"},
                "СП7":{"len":lens[7],"dup": count_dup[7],
                        "СП7": info_7[7],
                        "СП8": info_7[8],
                        "СП9": info_7[9],
                        "СП10":info_7[10],
                        "СП11":info_7[11],
                        "СП12":info_7[12],
                        "СП13":info_7[13],
                        "СП14":info_7[14],  
                        },
                "СП8":{"len":lens[8],"dup": count_dup[8],
                        "СП15": info_8[15],
                        "СП16": info_8[16],
                        "СП17": info_8[17],
                        "СП18": info_8[18],
                        "СП19": info_8[19],
                        "СП20": info_8[20],
                        "СП21": info_8[21],
                        },
                        }
        return summary
    
    def append(self, frame:pd.DataFrame, indx:int = -1, subindx:int = 0 ):
        """
            Конкантенировать строки из frame в справочник .pkl 
        """

        if indx in {5, 6}:
            f1, f2 = get_pkl_fname(indx, subindx)
            f1 = self.root_path / f1  # sp5, 6
            f2 = self.root_path / f2  # sp5,6_list
            source_frame_1 = dataframe_from_pkl(f1)
            source_frame_2 = dataframe_from_pkl(f2)
            self.logger.debug(f"append: frame = {frame.shape} source_frame_2.shape = {source_frame_2.shape}")
            if is_equal(source_frame_1, frame):
                source_frame_1 = pd.concat((source_frame_1, frame))
                dump_pkl(source_frame_1, f1)
                frame_list,errs = correct_frame_list(frame, source_frame_2)
                # self.update_sp_list(frame, indx)
                if len(errs) > 0:       
                    self.logger.debug(f'Ошибки изменения файла {f2}: {errs}')
                dump_pkl(frame_list, f2)
                print("prep_sp_lists start")
                if indx == 5:
                    prep_sp_lists_fast(f2, self.root_path / "sp5_list_prep.pkl", word_tokenize)
                elif indx == 6: 
                    prep_sp_lists_fast(f2, self.root_path / "sp6_list_prep.pkl", word_tokenize)
                #предобработка sp_list, all_table
                self.logger.debug(f"Предобработка файла {f2} завершена")
                return True
            return False
        elif indx in {3}:
            rows = list(frame.itertuples(index=False))
            self.dbase.insert_in_table("SP3", rows)
            return True
        else:
            f = self.root_path / get_pkl_fname(indx, subindx)
            source_frame = dataframe_from_pkl(f)
        if is_equal(source_frame, frame):
            source_frame = pd.concat((source_frame, frame))
            dump_pkl(source_frame, f)
            return True
        else:
            return False
        
    def replace(self,frame:pd.DataFrame, indx:int = -1, subindx:int = 0):
        """
            Заменить справочник
        """
        self.logger.info(f"Замена справочника sp{indx}_{subindx}.pkl frame: {frame.shape}")
        if indx == 5:
            fsp = self.root_path / get_pkl_fname(indx, subindx)[0]
            source_frame = dataframe_from_pkl(fsp)
            if is_equal(source_frame, frame):
                dump_pkl(frame, fsp)
                prep_sp_lists_fast(Path(path2directory()) /"sp5_list.pkl", #
                                   self.root_path / "sp5_list_prep.pkl", word_tokenize)
                self.logger.info("Предобработка из файла sp5_list.pkl в sp5_list_prep.pkl выполнена")
                return True
            else:
                return False
        if indx == 6:
            fsp = self.root_path / get_pkl_fname(indx, subindx)[0]
            source_frame = dataframe_from_pkl(fsp)
            if is_equal(source_frame, frame):
                dump_pkl(frame, fsp)
                prep_sp6_tables(path2directory(),"sp6_all_tables.pkl")
                self.logger.info("Сбор файлов в sp6_all_tables.pkl выполнен")
                prep_sp_lists_fast(Path(path2directory()) /"sp6_list.pkl", #
                                   self.root_path / "sp6_list_prep.pkl", word_tokenize)
                self.logger.info("Предобработка из файла sp6_list.pkl в sp6_list_prep.pkl выполнена")

                #prep_sp_lists?
                return True
            else:
                return False
        else:
            f = self.root_path / get_pkl_fname(indx, subindx)
            self.logger.info(f"Замена по пути {f}")
            source_frame = dataframe_from_pkl(f)
        if is_equal(source_frame, frame):
            dump_pkl(frame, f)
            return True
        else:
            return False
        
class SpDbManager:
    """
        Класс для работы со справочниками через базу данных
    """
    def __init__(self, path2sp: Union[str, Path], dbmanager):
        self.root_path = Path(path2sp)
        self.dbase = dbmanager
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler('sp_manager_log.txt')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info("SpDbmanager create")

    def save_csv(self, spname)->str:
        """"
            Сохранить в csv справочник
            return:
            fsave - path to csv
        """
        self.logger.debug(f"Получить справочник {spname}")
        dataframe = self.dbase.get_table(spname)
        fsave = self.root_path / (spname + ".csv")
        with open(fsave,'wb') as handler:
            dataframe.to_csv(fsave, index=False, sep=";", encoding="cp1251")
        return fsave
    
    def save_pattern(self, spname):
        """"
            Сохранить csv файл шаблон структуры справочника,
            return:
            fsave - path to csv
        """
        self.logger.debug(f"Получить шаблон-справочника {spname}")  
        columns = self.dbase.get_columns(spname)
        fsave = self.root_path / (spname + ".csv")
        with open(fsave, 'wb') as handler:
            df = pd.DataFrame(columns=columns)
            df.to_csv(fsave, index=False, sep=";", encoding="cp1251")
        return fsave

    def append(self, frame: pd.DataFrame, spname: str):
        """
            Конкантенировать строки из frame в справочник spname
            Добавляет только уникальные записи из frame, которых нет в БД
        """
        if spname in ["SP5LIST","SP6LIST"]:
            # frame = prep_sp_lists_fast(frame, word_tokenize)
            frame = prep_sp_lists(frame)
            self.logger.debug(f"{spname} обработан для вставки и БД")
        rows = list(frame.itertuples(index=False))
        try:
            self.logger.debug(f"Конкантенировать строки в справочник {spname}")
            self.dbase.insert_in_table(spname, rows)
        except:
            self.logger.debug(f"Неудалось конкантенировать {spname} shape={frame.shape}")
        return True

    def summary(self):
        """
            Краткая статистика по справочникам в системе
        """
        lens = {}
        count_dup = {}
        for spname in ["SP1","SP2","SP34","SP5","SP5LIST","SP7","SP8",
                       "SP6LIST","SP61","SP62","SP63", "SP64","SP65","SP66","SP67", "SP68","SP69", "SP610", "SP611", "SP612",]:
            table = self.dbase.get_table(spname)
            lens[spname] = len(table)
            count_dup[spname] = int(table.duplicated().sum())
            if spname in {"SP7"}:
                count_7 = table.groupby("номер_справочника").count()
                info_7 = count_7.to_dict()['Заголовок дела']
            if spname in {"SP8"}:
                count_8 = table.groupby("номер_справочника").count()
                info_8 = count_8.to_dict()['Заголовок дела']
        
        summary = {"СП1":   {"len":lens["SP1"],"dup": count_dup["SP1"]},
                   "СП2":   {"len":lens["SP2"],"dup": count_dup["SP2"]},
                   "СП34":  {"len":lens["SP34"],"dup": count_dup["SP34"]},
                   "СП5":   {"len":lens["SP5"],"dup": count_dup["SP5"]},
                   "СП5LIST":{"len": lens["SP5LIST"],"dup":count_dup["SP5LIST"]},
                   "СП61":   {"len": lens["SP61"],"dup":count_dup["SP61"]},
                   "СП62":   {"len": lens["SP62"],"dup":count_dup["SP62"]},
                   "СП63":   {"len": lens["SP63"],"dup":count_dup["SP63"]},
                   "СП64":   {"len": lens["SP64"],"dup":count_dup["SP64"]},
                   "СП65":   {"len": lens["SP65"],"dup":count_dup["SP65"]},
                   "СП66":   {"len": lens["SP66"],"dup":count_dup["SP66"]},
                   "СП67":   {"len": lens["SP67"],"dup":count_dup["SP67"]},
                   "СП68":   {"len": lens["SP68"],"dup":count_dup["SP68"]},
                   "СП69":   {"len": lens["SP69"],"dup":count_dup["SP69"]},
                   "СП610":   {"len": lens["SP610"],"dup":count_dup["SP610"]},
                   "СП611":   {"len": lens["SP611"],"dup":count_dup["SP611"]},
                   "СП612":   {"len": lens["SP612"],"dup":count_dup["SP612"]},
                   "СП6LIST":{"len": lens["SP6LIST"],"dup":count_dup["SP6LIST"]},  
                "СП7":{"len":lens["SP7"],"dup": count_dup["SP7"],
                        "СП7": info_7['7'],
                        "СП8": info_7['8'],
                        "СП9": info_7['9'],
                        "СП10":info_7['10'],
                        "СП11":info_7['11'],
                        "СП12":info_7['12'],
                        "СП13":info_7['13'],
                        "СП14":info_7['14'],  
                        },
                "СП8":{"len":lens["SP8"],"dup": count_dup["SP8"],
                        "СП15": info_8['15'],
                        "СП16": info_8['16'],
                        "СП17": info_8['17'],
                        "СП18": info_8['18'],
                        "СП19": info_8['19'],
                        "СП20": info_8['20'],
                        "СП21": info_8['21'],
                        },
                        }
        return summary
    

    def replace(self,frame:pd.DataFrame, spname:str):
        """
            Заменить справочник
        """
        self.logger.info(f"Замена справочника {spname} frame: {frame.shape}")
        source_frame = self.dbase.get_table(spname)
        print(source_frame.shape, frame.shape)

        if is_equal(source_frame,frame):
            self.dbase.clear_table(spname)
            if spname in ["SP5LIST","SP6LIST"]:
                # frame = prep_sp_lists(frame, word_tokenize)
                frame = prep_sp_lists(frame)
                self.logger.debug(f"{spname} обработан для вставки и БД")
        
            rows = list(frame.itertuples(index=False))
            self.dbase.insert_in_table(spname, rows)
            return True
        else:
            return False

def dataframe_from_pkl(fpath:str)->pd.DataFrame:
    with open(fpath, 'rb') as f:
        dataframe = pkl.load(f)
    return dataframe

def is_equal(frame,other):
    """
        Сравнить frame справочника на эквивалентность
    """
    print(frame.columns, other.columns)
    if isinstance(frame, pd.DataFrame):
        if len(frame.columns) != len(other.columns):
            return False
        if all(frame.columns == other.columns):
            return True
        else:
            return False
    else:
        return True

def dump_pkl(frame, path2save="."):
    with open(path2save,'wb') as file:
        pickle.dump(frame, file)

def get_dir_list(backups_dir="."):
    # утилита, возвращающая список каталогов с сохранениями
    return [e for e in os.listdir(backups_dir) if os.path.isdir(os.path.join(backups_dir, e))]


def get_files_in_dir(dir="."):
    # утилита, возвращающая список файлов в каталоге
    dir_list = os.listdir(dir)
    dir_list = [x for x in dir_list if x[-2:] == "db"]
    return dir_list


def make_backup(dir_from=".", dir_to="."):
    # создать backup
    import datetime as dt
    date = dt.datetime.today()
    files_in =  Path(dir_from).glob("*.db")
    dir_name = f"backup_{date.year}_{date.month}_{date.day}_{date.hour}_{date.minute}"
    os.mkdir(os.path.join(dir_to, dir_name))
    for file in files_in:
        shutil.copy2(os.path.join(dir_from, file), os.path.join(dir_to, dir_name))
    return dir_name

def copy_pkl_files(from_dir, to_dir):
    """
        copy pkl files from_dir to to_dir
    """
    for p in Path(to_dir).glob("*.pkl"):
        if p.is_file():
            p.unlink()
    for file_path in Path(from_dir).glob("*.pkl"):
        shutil.copyfile(file_path, to_dir + file_path.name)

def get_csv(number=1, sub_number=0, dir_from=".", dir_to="."):
    # утилита, возвращающая csv по номеру (и разделу) справочника
    dir_list = os.listdir(dir_from)
    dir_list = [x for x in dir_list if x[-4:] == ".pkl"]
    if sub_number == 0:
        if number == 4:
            f_name = f"sp{number}_proc.pkl"
        else:
            f_name = f"sp{number}.pkl"
    else:
        f_name = f"sp{number}_{sub_number}.pkl"
    
    if f_name in dir_list:
        with open(os.path.join(dir_from, f_name), 'rb') as f:
            dataframe = pkl.load(f)
            df = pd.DataFrame(columns=dataframe.columns)
            df.to_csv(os.path.join(dir_to, f_name[:-4] + ".csv"), index=False, sep=";", encoding="cp1251")
    else:
        print("Некорректный номер")
        return 1
    return 0


def clear_sent(text):


    stop_words = stopwords.words('russian')

    def get_tocken(text):
        '''Вспомогательная функция для очистки от стоп слов'''
        text = re.sub('[/\\\]', ' ', text)
        tokens = word_tokenize(text)
        lemma_transform = WordNetLemmatizer()
        tokens = [lemma_transform.lemmatize(token) for token in tokens if token not in stop_words]
        return Counter(tokens)

    return get_tocken(text)


def get_similarity(text1, text2):

    def cosine_simliarity(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = np.sqrt(sum1) * np.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
    return cosine_simliarity(clear_sent(text1), clear_sent(text2))


def get_cleared_table(work_dir, file_name):

    with open(os.path.join(work_dir, file_name), 'rb') as f:
        df = pkl.load(f)
    cleared_table_list = []
    # найти столбец с наибольшим количеством букв
    s_max = 0
    col_max = ""
    for col in df.columns:
        s = df[col].apply(lambda x: len(x)).sum()
        if s > s_max:
            s_max = s
            col_max = col
    return [clear_sent(x.lower()) for x in df[col_max]]


ALLOWED_EXTENSIONS = {'pdf', 'csv', 'xls', 'xlsx'}

def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def bool2str(x):
    if isinstance(x, bool):
        if x == True:
            return "Верно"
        else:
            return "Неверно"
    else:
        return x


def create_empty_ans(fname):
    ans = {
        0: {
            "name": "Проверка №0: Шаблон таблицы",
            "status": False,
            "coment": f"Не удалось проверить корректность таблицы в  файле:{fname}",
        },
        1: {"name": names_checks[1], "status": None, "coment": ""},
        2: {"name": names_checks[2], "status": None, "coment": ""},
        3: {"name": names_checks[3], "status": None, "coment": ""},
        4: {"name": names_checks[4], "status": None, "coment": ""},
        5: {"name": names_checks[5], "status": None, "coment": ""},
        6: {"name": names_checks[6], "status": None, "coment": ""},
        7: {"name": names_checks[7], "status": None, "coment": ""},
        8: {"name": names_checks[8], "status": None, "coment": ""},
        9: {"name": names_checks[9], "status": None, "coment": ""},
        10: {"name": names_checks[10], "status": None, "coment": ""},
        11: {"name": names_checks[11], "status": None, "coment": ""},
        12: {"name": names_checks[12], "status": None, "coment": ""},
        13: {"name": names_checks[13], "status": None, "coment": ""},
        14: {"name": names_checks[14], "status": None, "coment": ""},
        15: {"name": names_checks[15], "status": None, "coment": ""},
        16: {"name": names_checks[16], "status": None, "coment": ""},
        17: {"name": names_checks[17], "status": None, "coment": ""},
        18: {"name": names_checks[18], "status": None, "coment": ""},
        19: {"name": names_checks[19], "status": None, "coment": ""},
        20: {"name": names_checks[20], "status": None, "coment": ""},
        21: {"name": names_checks[21], "status": None, "coment": ""},
        22: {"name": names_checks[22], "status": None, "coment": ""},
        23: {"name": names_checks[23], "status": None, "coment": ""},
        24: {"name": names_checks[24], "status": None, "coment": ""},
        25: {"name": names_checks[25], "status": None, "coment": ""},
        26: {"name": names_checks[26], "status": None, "coment": ""},
        27: {"name": names_checks[27], "status": None, "coment": ""},
        28: {"name": names_checks[28], "status": None, "coment": ""},
    }
    for k in ans:
        ans[k]["status"] = str(ans[k]["status"])
    return ans