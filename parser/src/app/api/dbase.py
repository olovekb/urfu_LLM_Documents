import sqlite3
import pandas as pd

class DBManager:
    def __init__(self, path2db) -> None:
        self.path2db = path2db
        self.connection = sqlite3.connect(self.path2db)
        self.table_names = ["SP1","SP2","SP34","SP5","SP5LIST",
                            "SP61","SP62","SP63", "SP64","SP65","SP66","SP67", "SP68","SP69", "SP610", "SP611", "SP612",
                            "SP6LIST","SP7","SP8"]
    
    def get_table(self,table_name):
        """
            Получить таблицу из базы в pd.DataFrame формате
        """
        if table_name in self.table_names:
            cols = [x[1] for x in self.execute_query(f"PRAGMA table_info({table_name})")]
            rows_lines = self.execute_query(f"SELECT * From {table_name}")
            table = pd.DataFrame.from_records(data = rows_lines, columns=cols)
            return table
        else:
            return AssertionError(f"Таблица {table_name} отсутствует в БД")
    
    def get_columns(self, table_name):
        if table_name in self.table_names:
            cols = [x[1] for x in self.execute_query(f"PRAGMA table_info({table_name})")]
            return cols
        else:
            return AssertionError(f"Таблица {table_name} отсутствует в БД")
        
    def insert_in_table(self, table_name, rows):
        if table_name in self.table_names:
            s = "?,"*len(rows[0])
            s = s[:-1]
            query = f"INSERT INTO {table_name} VALUES ({s}) ON CONFLICT DO NOTHING"
            try:
                with sqlite3.connect(self.path2db) as conn:
                    cursor = conn.cursor()
                    cursor.executemany(query, rows)
                    conn.commit()
            except Exception as e:
                return AssertionError(f"Ошибка при вставке в {table_name}:{e}")
        else:
            return AssertionError(f"Таблица {table_name} отсутствует в БД")

    def clear_table(self, table_name):
        """
            Очистить таблицу table_name
        """
        query = f"DELETE FROM {table_name};"
        self.execute_query(query)
        
    def execute_query(self, query, params=None):
        """Метод для выполнения SQL-запроса."""
        try:
            with sqlite3.connect(self.path2db) as conn:
                cursor = conn.cursor()
                if params is None:
                    cursor.execute(query)
                else:
                    cursor.execute(query, params)
                return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Ошибка при работе с SQLite: {e}")
        