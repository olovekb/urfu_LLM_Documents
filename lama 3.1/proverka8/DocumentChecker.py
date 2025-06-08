import re
import cv2
import pytesseract
import poppler
import fitz  # PyMuPDF
import os
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util

"""
# Загрузка модели эмбеддингов
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Сравнение с использованием эмбеддингов напрямую
def compare_names_with_embeddings(name1, name2, threshold=0.8):
    if name1 and name2:
        name1_clean = preprocess_text(name1)
        name2_clean = preprocess_text(name2)
        emb1 = model.encode(name1_clean, convert_to_tensor=True)
        emb2 = model.encode(name2_clean, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()

        if similarity >= threshold:
            return f"Наименования совпадают (сходство: {similarity:.2f})"
        else:
            return f"Наименования не совпадают (сходство: {similarity:.2f})"
    else:
        return "Ошибка: Не удалось извлечь наименования"
"""
#Преобразование pdf к jpg
def convertPDFToImage(file_path, poppler_path = 'D:\\OlegDocAnalyze\\fork_urfu_LLM_DOC\\urfu_LLM_Documents\\lama 3.1\\poppler-24.08.0\\Library\\bin'):
    deleteFileInFolder()
    os.environ["PATH"] += os.pathsep + poppler_path
    images = convert_from_path(file_path)

    for i, image in enumerate(images, start=1):
        image.save(f'D:\\OlegDocAnalyze\\fork_urfu_LLM_DOC\\urfu_LLM_Documents\\lama 3.1\\proverka8\\data\\pdf_images\\page_{i}.jpg', 'JPEG')

#Удаление предыдущих результатов преобразования pdf к jpg
def deleteFileInFolder(file_path = 'D:\\OlegDocAnalyze\\fork_urfu_LLM_DOC\\urfu_LLM_Documents\\lama 3.1\\proverka8\\data\\pdf_images'):
    for filename in os.listdir(file_path):
        file_path = os.path.join(file_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')

#Анализ страницы
def analyzeImages(images_path = 'D:\\OlegDocAnalyze\\fork_urfu_LLM_DOC\\urfu_LLM_Documents\\lama 3.1\\proverka8\\data\\pdf_images'):
    pytesseract.pytesseract.tesseract_cmd = 'D:\\OlegDocAnalyze\\fork_urfu_LLM_DOC\\urfu_LLM_Documents\\lama 3.1\\Tesseract\\tesseract.exe'

    #Открываем контекст одной страницы
    for i in range(1, len(os.listdir(images_path)) + 1):
        img = cv2.imread(f'D:\\OlegDocAnalyze\\fork_urfu_LLM_DOC\\urfu_LLM_Documents\\lama 3.1\\proverka8\\data\\pdf_images\\page_{i}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(pytesseract.image_to_string(img, lang='rus'))


# Основная логика
def main(pdf_path):
    convertPDFToImage(pdf_path)
    return f"Images successfully converted from {pdf_path}"

# Пример
pdf_path = 'D:\\OlegDocAnalyze\\fork_urfu_LLM_DOC\\urfu_LLM_Documents\\lama 3.1\\proverka8\\data\\Агаповский_архив,_КСП,_ф_79,_оп_2_л_с_за_2022_год (2).pdf'
main(pdf_path)
analyzeImages()