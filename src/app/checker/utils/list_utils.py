from tabula import read_pdf
import pandas as pd


def get_tables(path: str):
    return read_pdf(path, pages="all")


def read_sp1(PATH, f_name):
    """
        Прочитать список СП1
    """
    df = pd.read_excel(PATH + f_name).fillna('')
    df = df[df[df.columns[0]] != ''].reset_index(drop=True)
    df.columns = df.iloc[0]
    data = df.iloc[1:].reset_index(drop=True)
    return data


def read_sp2(PATH, f_name):
    """
        Прочитать список СП2
    """
    dF = pd.read_excel(PATH + f_name)
    data = dF.iloc[:,1]
    data = data[2:]
    data = data.reset_index(drop=True)
    return data


def read_sp3(PATH, f_name):
    """
        Прочитать список СП3
    """
    data = pd.read_excel(PATH + f_name)
    data = data.iloc[0:,1:]
    data = data.rename(columns=data.iloc[1,:])
    data = data.iloc[2:,:]
    data = data.reset_index(drop=True)
    data.insert(2, "Архивное подразделение", ['']*data.shape[0])
    data.insert(3, "Наименование подразделения", ['']*data.shape[0])

    # убрать заголовки и внести их в таблицу
    arch_val = ""
    dep_val = ""
    drops = []
    for row in range(data.shape[0]):
        d = data.iloc[row, :][0]
        if "Архив:" in d:
            arch_val = d[d.find(':')+1:]
            drops.append(row)
        if "Наименование:" in d:
            dep_val = d[d.find(':')+1:]
            drops.append(row)
        data.at[row,"Архивное подразделение"] = arch_val
        data.at[row,"Наименование подразделения"] = dep_val
    data = data.reset_index(drop=True)
    drops.append(data.shape[0]-1)
    drops = list(set(drops))
    for row in drops:
        data = data.drop([row])
    data = data.reset_index(drop=True)
    return data


def read_sp4(PATH, f_name):
    """
        Прочитать список СП4
    """
    dF = get_tables(PATH + f_name)
    df = pd.DataFrame(dF[0])
    df = df.fillna(method='bfill')
    df_mod = df[~(df['Архив'].str.contains('Наименование') | df['Архив'].str.contains('фонд'))]
    df_mod = df_mod.groupby(['Наименование'], as_index = False).first().reset_index(drop=True)
    for col in df_mod.columns:
        df_mod[col] = pd.Series(map(lambda x: x.replace('\r', ' ').replace('\n', ' '), df_mod[col].apply(str)))
    return df_mod