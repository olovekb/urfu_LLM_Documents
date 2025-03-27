
import unittest
import requests
from pathlib import Path
from utils import get_pkl_fname
from io import BytesIO
import pandas as pd


path2files = Path(".\\src\\app\\data\\work_lists\\")
class api_test(unittest.TestCase):
    # def test_uploade_guide(self):
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "1.xls", 'rb')},data={'id':1})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "2.xls", 'rb')},data={'id':2})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "3.xls", 'rb')},data={'id':3})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "4.pdf", 'rb')},data={'id':4})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "5.pdf", 'rb')},data={'id':5})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "6.pdf", 'rb')},data={'id':6})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "7.pdf", 'rb')},data={'id':7})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "8.pdf", 'rb')},data={'id':8})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "9.pdf", 'rb')},data={'id':9})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "10.pdf", 'rb')},data={'id':10})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "11.pdf", 'rb')},data={'id':11})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "12.pdf", 'rb')},data={'id':12})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "13.pdf", 'rb')},data={'id':13})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "14.pdf", 'rb')},data={'id':14})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "15.pdf", 'rb')},data={'id':15})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "16.pdf", 'rb')},data={'id':16})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "17.pdf", 'rb')},data={'id':17})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "18.pdf", 'rb')},data={'id':18})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "19.pdf", 'rb')},data={'id':19})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "20.pdf", 'rb')},data={'id':20})
    #     r = requests.post(url = "http://localhost:8005/update_guide/", files = {'file': open(path2files / "21.pdf", 'rb')},data={'id':21})

    def test_get_guide(self):
        for num in range(1,22):
            for sub_num in range(0,3):
                if num in [7,15,16,17,19]:
                    if sub_num > 0:
                        ans = requests.get(url = "http://localhost:8005/guide/",params={'id':num, "subid":sub_num})
                        if ans.status_code != 404:
                            stream = BytesIO(ans.content)
                            df = pd.read_excel(stream, engine='openpyxl')
                            file = get_pkl_fname(num, sub_num)
                            name = file.split(".")[0]
                            df.to_excel(f"xlsx\\{name}.xlsx",index=False)
                else:
                    if sub_num < 1:
                        ans = requests.get(url = "http://localhost:8005/guide/",params={'id':num, "subid":sub_num})
                        if ans.status_code != 404:
                            stream = BytesIO(ans.content)
                            df = pd.read_excel(stream, engine='openpyxl')
                            file = get_pkl_fname(num, sub_num)
                            name = file.split(".")[0]
                            df.to_excel(f"xlsx\\{name}.xlsx",index=False)
                print(num,sub_num, ans.status_code)
           

if __name__ == '__main__':
    unittest.main()