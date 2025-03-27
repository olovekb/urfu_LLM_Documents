import unittest
from pathlib import Path
import pandas as pd
import sys
import os
import tracemalloc
import asyncio
import numpy as np


# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.append(os.getcwd())
from src.app.api.main import check_document, ner_pipeline
from src.app.api.utils import path2directory


def get_status(frame, name):
    status = {}
    sub = frame[frame.name == name][["number", "status"]]
    for k, row in sub.iterrows():
        status[int(row["number"])] = row["status"]
    return status


class api_test(unittest.TestCase):

    def test_documents(self):
        asyncio.run(self.async_test_documents())

    async def async_test_documents(self):
        path2pdf = Path(
            os.getcwd() + "/src/app/data/test/"
        )  # папка с тестовыми описями
        res_frame = pd.read_excel(path2pdf / "Статусы.xlsx")

        files = list(path2pdf.glob("*pdf"))
        for file in files:
            fname = file.stem
            print(fname)
            ans = await check_document(file, ner_pipeline,0)
            res_status = get_status(res_frame, fname)
            print(ans)
            all_status = []
            for num_ch, body in ans.items():
                all_status += [body["status"] == str(res_status[num_ch])]
                print(num_ch, " ", body["status"], res_status[num_ch])

            all_status = np.array(all_status)
            print(all_status, all(all_status == True), not (all(all_status == True)))
            if all(all_status == True) == False:
                index = np.where(all_status == False)[0].astype(str)
                print(index)
                self.assertEqual(
                    all(all_status == True),
                    True,
                    f"Непрошли проверки №{','.join(index)} в {fname}",
                )
            else:
                print("s=", all(all_status == True))


if __name__ == "__main__":
    unittest.main()
