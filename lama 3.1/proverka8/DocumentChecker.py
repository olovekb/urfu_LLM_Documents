import cv2
import pytesseract
import os
from pathlib import Path
from pdf2image import convert_from_path
from sklearn.cluster import DBSCAN

def local_to_absolute_path(file_path):
    return str(Path(file_path).resolve())

# Очищаем папку от старых файлов
def clear_folder(folder_path):
    # Проверка, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Папка не найдена: {folder_path}")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Удаление файла
                print(f"Удален файл: {file_path}")
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}: {e}')

# Преобразование pdf к jpg
def convertPDFToImage(file_path, save_image_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images'), poppler_path = local_to_absolute_path('urfu_LLM_Documents/lama 3.1/poppler-24.08.0/Library/bin')):
    deleteFileInFolder()
    os.environ["PATH"] += os.pathsep + poppler_path
    images = convert_from_path(file_path)

    for i, image in enumerate(images, start=1):
        image.save(f'{save_image_path}\\page_{i}.jpg', 'JPEG')

# Удаление предыдущих результатов преобразования pdf к jpg
def deleteFileInFolder(file_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images')):
    for filename in os.listdir(file_path):
        file_path = os.path.join(file_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')

# Анализ страницы
def analyzeImages(images_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images')):
    pytesseract.pytesseract.tesseract_cmd = local_to_absolute_path('lama 3.1/proverka8/Tesseract/tesseract.exe')

    # Открываем контекст одной страницы
    for i in range(1, len(os.listdir(images_path)) + 1):
        img = cv2.imread(f'{local_to_absolute_path("lama 3.1/proverka8/data/pdf_images")}\\page_{i}.jpg')

        data = pytesseract.image_to_data(img, lang='rus', output_type=pytesseract.Output.DICT)

        boxes = []
        centers = []

        # 0 - утверждено, 1 - согласовано
        flag_arr = [False, False]
        left_arr = [0, 0]
        top_arr = [0, 0]

        # Сбор прямоугольников для текста
        for iterator in range(len(data['text'])):

            # Условие для поиска нужных наименований
            if 'утверждаю' in data['text'][iterator].lower().strip() or 'утверждено' in data['text'][iterator].lower().strip():
                flag_arr[0] = True
                left_arr[0] = int(data['left'][iterator])
                top_arr[0] = int(data['top'][iterator])

            if 'согласовано' in data['text'][iterator].lower().strip():
                flag_arr[1] = True
                left_arr[1] = int(data['left'][iterator])
                top_arr[1] = int(data['top'][iterator])

            # Сбор прямоугольников для блоков с наименованиями
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

            # Рисуем прямоугольники вокруг блоков текста
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (10, 255, 12), 2)

            # Извлечение текста внутри рамки
            cropped_img = img[y_min:y_max, x_min:x_max]
            text = pytesseract.image_to_string(cropped_img, lang='rus')

            if text.strip():  # Если текст не пустой, выводим его
                print(f"Текст внутри рамки на странице {i}:")
                print(text)
                print("\n")

        # Сохраняем изображение с выделенными блоками
        cv2.imwrite(f'{local_to_absolute_path("lama 3.1/proverka8/data/change_image")}\\page_{i}.jpg', img)

# Основная логика
def main(pdf_path):
    # Очищаем папки pdf_images и change_image перед запуском
    clear_folder(local_to_absolute_path('lama 3.1/proverka8/data/pdf_images'))
    clear_folder(local_to_absolute_path('lama 3.1/proverka8/data/change_image'))

    convertPDFToImage(pdf_path)
    analyzeImages()
    return f"Images successfully converted from {pdf_path}"

# Запуск
pdf_path = r'G:/Python/urfu_LLM_documents/lama 3.1/proverka8/data/Еткуль, УСЗН, оп 1, п.хр.2021.pdf'

main(pdf_path)
