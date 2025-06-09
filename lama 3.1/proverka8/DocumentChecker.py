import re
import cv2
import pytesseract
import poppler
import fitz  # PyMuPDF
import os
import numpy as np
from certifi import where
from cv2.gapi import kernel
from networkx.algorithms.bipartite.cluster import clustering
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN

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
def convertPDFToImage(file_path, poppler_path = 'G:\\Python\\urfu_LLM_documents\\lama 3.1\\poppler-24.08.0\\Library\\bin'):
    deleteFileInFolder()
    os.environ["PATH"] += os.pathsep + poppler_path
    images = convert_from_path(file_path)

    for i, image in enumerate(images, start=1):
        image.save(f'G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\pdf_images\\page_{i}.jpg', 'JPEG')

#Удаление предыдущих результатов преобразования pdf к jpg
def deleteFileInFolder(file_path = 'G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\pdf_images'):
    for filename in os.listdir(file_path):
        file_path = os.path.join(file_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')

#Анализ страницы
def analyzeImages(images_path = 'G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\pdf_images', template_path = 'G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\template_images'):
    pytesseract.pytesseract.tesseract_cmd = 'G:\\Python\\urfu_LLM_documents\\lama 3.1\\Tesseract\\tesseract.exe'

    template_array = []
    for j in range(1,len(os.listdir(template_path)) + 1):
        img = cv2.imread(
            f'G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\template_images\\page_{j}.png')
        template_array.append(img)

    results_array = []
    text_on_page = ''

    #Открываем контекст одной страницы
    for i in range(1, len(os.listdir(images_path)) + 1):
        img = cv2.imread(f'G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\pdf_images\\page_{i}.jpg')

        #Выделение блоков на изображении
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
        dilate = cv2.dilate(thresh, kernal, iterations=1)


        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            #if h > 100 and w > 200:
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.imwrite(f'G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\change_image\\page_{i}.jpg', img)


        """
        text_on_page = pytesseract.image_to_string(img, lang='rus').lower()
        print(text_on_page)

        if ('утверждаю' in text_on_page or 'согласовано' in text_on_page):
            print('Найден искомый текст!')

            for template in template_array:
                gray_main = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                # Шаг 4: Сопоставление шаблонов
                result = cv2.matchTemplate(gray_main, gray_template, cv2.TM_CCOEFF)

                # Шаг 5: Поиск лучшего совпадения
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                best_match_location = max_loc

                # Шаг 6: Отрисовка прямоугольника
                h, w = gray_template.shape
                bottom_right = (best_match_location[0] + w, best_match_location[1] + h)
                cv2.rectangle(img, best_match_location, bottom_right, (0, 255, 0), 2)

                # Шаг 7: Отображение результата
                #cv2.imshow('Результат', img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            """

# Основная логика
def main(pdf_path):
    convertPDFToImage(pdf_path)
    return f"Images successfully converted from {pdf_path}"

# Пример
pdf_path = 'G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\Агаповский_архив,_КСП,_ф_79,_оп_2_л_с_за_2022_год (2).pdf'
main(pdf_path)
analyzeImages()