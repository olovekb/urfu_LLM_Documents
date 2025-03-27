import os
import sys
sys.path.append("/media/kirilman/Z2/Project/Sber_document/documents/src/")
sys.path.append("./")
print(sys.path)
from src.app.checker.utils.text_utils import read_text_documents
p = "src/app/data/test/Миасс,_геологоразведочный_колледж,_оп_1_пост.pdf"
text, full_text, page_blocks = read_text_documents(p)
print(page_blocks)