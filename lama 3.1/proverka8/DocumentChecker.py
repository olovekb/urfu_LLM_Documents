import cv2
import pytesseract
import os
from cv2.gapi import kernel
from pdf2image import convert_from_path
from sklearn.cluster import DBSCAN
from pathlib import Path

def local_to_absolute_path(file_path):
    return str(Path(file_path).resolve())

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
def convertPDFToImage(file_path, save_image_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images'), poppler_path = local_to_absolute_path('urfu_LLM_Documents/lama 3.1/poppler-24.08.0/Library/bin')):
    deleteFileInFolder()
    os.environ["PATH"] += os.pathsep + poppler_path
    images = convert_from_path(file_path)

    for i, image in enumerate(images, start=1):
        image.save(f'{save_image_path}\\page_{i}.jpg', 'JPEG')

#Удаление предыдущих результатов преобразования pdf к jpg
def deleteFileInFolder(file_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images')):
    for filename in os.listdir(file_path):
        file_path = os.path.join(file_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')

#Анализ страницы
def analyzeImages(images_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images')):
    pytesseract.pytesseract.tesseract_cmd = local_to_absolute_path('lama 3.1/proverka8/Tesseract/tesseract.exe')

    #Открываем контекст одной страницы
    for i in range(1, len(os.listdir(images_path)) + 1):
        img = cv2.imread(f'{local_to_absolute_path('lama 3.1/proverka8/data/pdf_images')}\\page_{i}.jpg')

        data = pytesseract.image_to_data(img, lang='rus', output_type=pytesseract.Output.DICT)

        boxes = []
        centers = []

        #0 - утверждено, 1 - согласовано
        flag_arr = [False, False]
        left_arr = [0, 0]
        top_arr = [0, 0]

        # Сбор прямоугольников для текста
        for iterator in range(len(data['text'])):

            if 'утверждаю' in data['text'][iterator].lower().strip() or 'утверждено' in data['text'][iterator].lower().strip():
                flag_arr[0] = True
                left_arr[0] = int(data['left'][iterator])
                top_arr[0] = int(data['top'][iterator])

            if 'согласовано' in data['text'][iterator].lower().strip():
                flag_arr[1] = True
                left_arr[1] = int(data['left'][iterator])
                top_arr[1] = int(data['top'][iterator])

            if flag_arr[0]:
                if int(data['conf'][iterator]) > 50 and flag_arr[0] and data['text'][iterator] != None and (left_arr[0] - 250 <= data['left'][iterator] or left_arr[0] + 250 >= data['left'][iterator]) and (top_arr[0] - 500 <= data['top'][iterator] and top_arr[0] + 500 >= data['top'][iterator]):
                    x, y, w, h = data['left'][iterator], data['top'][iterator], data['width'][iterator], data['height'][iterator]
                    cx, cy = x + w // 2, y + h // 2
                    boxes.append((x, y, w, h))
                    centers.append([cx, cy])

            if flag_arr[1]:
                if int(data['conf'][iterator]) > 50 and flag_arr[1] and data['text'][iterator] != None and (left_arr[1] - 250 <= data['left'][iterator] or left_arr[1] + 250 >= data['left'][iterator]) and (top_arr[1] - 500 <= data['top'][iterator] and top_arr[1] + 500 >= data['top'][iterator]):
                    x, y, w, h = data['left'][iterator], data['top'][iterator], data['width'][iterator], data['height'][iterator]
                    cx, cy = x + w // 2, y + h // 2
                    boxes.append((x, y, w, h))
                    centers.append([cx, cy])

        if not centers:
            print("Текст не найден.")
            continue

        # Кластеризация по координатам центров
        clustering = DBSCAN(eps=160, min_samples=1).fit(centers)
        labels = clustering.labels_

        # Группируем боксы по кластерам
        clusters = {}
        for iterator, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(boxes[iterator])

        # Рисуем рамки вокруг абзацев
        for box_list in clusters.values():
            xs = [x for (x, y, w, h) in box_list]
            ys = [y for (x, y, w, h) in box_list]
            ws = [x + w for (x, y, w, h) in box_list]
            hs = [y + h for (x, y, w, h) in box_list]

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(ws), max(hs)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (10, 255, 12), 2)

        cv2.imwrite(f'{local_to_absolute_path('lama 3.1/proverka8/data/change_image')}\\page_{i}.jpg', img)

        """
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

# Основная логика
def main(pdf_path):
    convertPDFToImage(pdf_path)
    analyzeImages()
    return f"Images successfully converted from {pdf_path}"

# Запуск
pdf_path = r'G:/Python/urfu_LLM_documents/lama 3.1/proverka8/data/СП_16_Магнит_отдел_Выборы_Гос_Дума_2016_оп_2_п_хр_.pdf'

main(pdf_path)
