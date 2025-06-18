import cv2
import pytesseract
import os
from cv2.gapi import kernel
from pdf2image import convert_from_path
from Levenshtein import ratio
from sklearn.cluster import DBSCAN
from Comparison import main as compare_names_with_faiss
from pathlib import Path
import json


def local_to_absolute_path(file_path):
    return str(Path(file_path).resolve())

def check_empty_array_elements(array):
    for i in range(len(array)):
        if array[i] == '' and array[i+1] == '' and array[i+2] == '' and array[i+3] == '':
            return False
    return True

# Сравнение с использованием эмбеддингов напрямую
def compare_names_with_embeddings(threshold=0.6):
    #print('Шаг 3 Анализ блоков \'Утверждаю\' и \'Согласовано\'')
    pytesseract.pytesseract.tesseract_cmd = local_to_absolute_path('Libraries/Tesseract/tesseract.exe')

    approve_arr = [] # утверждено/утверждаю
    agreed_arr = [] # согласовано 
    

    #todo нужно сделать отсеивание элементов, если они идут спустя 4 пустых элемента массива
    for value in os.listdir(local_to_absolute_path('proverka8/data/change_image')):
        img = cv2.imread(local_to_absolute_path('proverka8/data/change_image/' + value))
        data = pytesseract.image_to_data(img, lang='rus', output_type=pytesseract.Output.DICT)

        for iterator in range(len(data)):
            if 'утверждаю' in data['text'][iterator].lower().strip() or 'утверждено' in data['text'][iterator].lower().strip():
                buffer = str(' '.join(data['text']))
                #print(f"\n{'='*50}\n"
                    #f"Текст внутри рамки: УТВЕРЖДЕНО/УТВЕРЖДАЮ\n"
                    #f"{'='*50}")
                print(buffer)
                approve_arr.append(buffer)
                break

            elif 'согласов' in data['text'][iterator].lower().strip():
                buffer = ' '.join(data['text'][iterator:])
                print("\n")
                #print(f"\n{'='*50}\n"
                   # f"Текст внутри рамки: СОГЛАСОВАНО\n"
                    #f"{'='*50}")
                print(buffer)
                agreed_arr.append(buffer)
                break

    data = {
        "approve_arr": approve_arr,
        "agreed_arr": agreed_arr
    }
    
    with open('proverka8/data/BlockNames.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

#Преобразование pdf к jpg
def convert_PDF_to_image(file_path, save_image_path = local_to_absolute_path('proverka8/data/pdf_images'), poppler_path = local_to_absolute_path('urfu_LLM_documents/Libraries/poppler-24.08.0/Library/bin')):
    #print('Шаг 1 Конвертация pdf к изображению...')
    os.environ["PATH"] += os.pathsep + poppler_path
    images = convert_from_path(file_path)

    for i, image in enumerate(images, start=1):
        image.save(f'{save_image_path}\\page_{i}.jpg', 'JPEG')

#Удаление предыдущих результатов преобразования pdf к jpg
def delete_file_in_folder():

    file_path_to_delete = local_to_absolute_path('proverka8/data/pdf_images') + '\\'
    for value in os.listdir(file_path_to_delete):
        try:
            if os.path.isfile(file_path_to_delete + value):
                os.remove(file_path_to_delete + value)
        except Exception as e:
            print(f'Ошибка при удалении файла {value}. {e}')

    file_path_to_delete = local_to_absolute_path('proverka8/data/change_image')  + '\\'
    for value in os.listdir(file_path_to_delete):
        try:
            if os.path.isfile(file_path_to_delete + value):
                os.remove(file_path_to_delete + value)
        except Exception as e:
            print(f'Ошибка при удалении файла {value}. {e}')

#Анализ страницы
def get_text_blocks(images_path = local_to_absolute_path('proverka8/data/pdf_images')):
    #print('Шаг 2 Подготовка к обработке изображения')
    pytesseract.pytesseract.tesseract_cmd = local_to_absolute_path('Libraries/Tesseract/tesseract.exe')

    page_counter = 0

    #Открываем контекст одной страницы
    for i in range(1, len(os.listdir(images_path)) + 1):
        img = cv2.imread(f'{local_to_absolute_path('proverka8/data/pdf_images')}\\page_{i}.jpg')

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

            if 'согласов' in data['text'][iterator].lower().strip():
                flag_arr[1] = True
                left_arr[1] = int(data['left'][iterator])
                top_arr[1] = int(data['top'][iterator])

            # Для ширины удаление от края
            if flag_arr[0]:
                if int(data['conf'][iterator]) > 80 and data['text'][iterator] != None and (left_arr[0] - 250 <= data['left'][iterator] or left_arr[0] + 250 >= data['left'][iterator]) and (top_arr[0] - 250 <= data['top'][iterator] and top_arr[0] + 250 >= data['top'][iterator]):
                    x, y, w, h = data['left'][iterator], data['top'][iterator], data['width'][iterator], data['height'][iterator]
                    cx, cy = x + w // 2, y + h // 2
                    boxes.append((x, y, w, h))
                    centers.append([cx, cy])

            # Для высоты удаление от края
            if flag_arr[1]:
                if int(data['conf'][iterator]) > 80 and data['text'][iterator] != None and (left_arr[1] - 250 <= data['left'][iterator] or left_arr[1] + 250 >= data['left'][iterator]) and (top_arr[1] - 250 <= data['top'][iterator] and top_arr[1] + 250 >= data['top'][iterator]):
                    x, y, w, h = data['left'][iterator], data['top'][iterator], data['width'][iterator], data['height'][iterator]
                    cx, cy = x + w // 2, y + h // 2
                    boxes.append((x, y, w, h))
                    centers.append([cx, cy])

        if not centers:
            #print("Обработка в процессе...")
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

        cropped_image = img

        # Рисуем рамки вокруг абзацев
        for box_list in clusters.values():
            xs = [x for (x, y, w, h) in box_list]
            ys = [y for (x, y, w, h) in box_list]
            ws = [x + w for (x, y, w, h) in box_list]
            hs = [y + h for (x, y, w, h) in box_list]

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(ws), max(hs)

            page_counter += 1

            cropped_image = img[y_min - 30:y_max + 30, x_min - 30 :x_max + 30]

            cropped_image_data = pytesseract.image_to_data(cropped_image, lang='rus', output_type=pytesseract.Output.DICT)

            for iterator in range(len(cropped_image_data['text'])):
                if 'утверждаю' in cropped_image_data['text'][iterator].lower().strip() or 'утверждено' in cropped_image_data['text'][iterator].lower().strip() or 'согласов' in cropped_image_data['text'][iterator].lower().strip():
                    cv2.imwrite(f'{local_to_absolute_path('proverka8/data/change_image')}\\cropped_image_{page_counter}.jpg', cropped_image)
                    #print(f'Добавлен файл {local_to_absolute_path('proverka8/data/change_image')}\\cropped_image_{page_counter}.jpg')




# Основная логика
def main(pdf_path):
    delete_file_in_folder()
    convert_PDF_to_image(pdf_path)
    #print(f'Изображения успешно конвертированы из {pdf_path}')
    get_text_blocks()
    #print(f'Обработка завершена!')
    compare_names_with_embeddings()
    compare_names_with_faiss() # - запускает функцию main из файла Comparison.py

