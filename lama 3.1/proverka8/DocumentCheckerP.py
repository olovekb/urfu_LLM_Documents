import cv2
import numpy as np
import os
from cv2.gapi import kernel
from pdf2image import convert_from_path
from pathlib import Path

def local_to_absolute_path(file_path):
    return str(Path(file_path).resolve())

# Преобразование pdf к jpg
def convert_PDF_to_image(file_path, save_image_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images'), poppler_path = local_to_absolute_path('urfu_LLM_Documents/lama 3.1/poppler-24.08.0/Library/bin')): 
    print('Шаг 1 Конвертация pdf к изображению...')
    os.environ["PATH"] += os.pathsep + poppler_path
    images = convert_from_path(file_path)

    for i, image in enumerate(images, start=1):
        image.save(f'{save_image_path}\\page_{i}.jpg', 'JPEG')

# Построение таблицы
def build_table(images_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images')):

    for iterator in range(1, len(os.listdir(images_path)) + 1):
        image_path = local_to_absolute_path('lama 3.1/proverka8/data/pdf_images') + f'\\page_{iterator}.jpg'
        image = cv2.imread(image_path)
        
        # Проверка, что изображение успешно загружено
        if image is None:
            print(f"Не удалось загрузить изображение {image_path}")
            continue  # Переход к следующему изображению, если не удалось загрузить текущее
        
        filterd_image = cv2.medianBlur(image, 1)
        gray = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)

        # Бинаризация
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Поиск контуров заголовков
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_contours = np.uint8(np.zeros((image.shape[0], image.shape[1])))

        cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)

        # Найдём прямоугольники заголовков
        headers = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 80 and h > 30:  # Фильтрация по размеру
                headers.append((x, y, w, h))

        # Сортировка по координате X (влево-направо)
        headers = sorted(headers, key=lambda b: b[0])

        # Определение вертикальных линий
        for x, y, w, h in headers:
            start_point = (x, y)
            end_point = (x, image.shape[0])  # до низа изображения
            cv2.line(image, start_point, end_point, (0, 0, 0), 2)

        # Горизонтальные линии (примерно на основе первого заголовка)
        if headers:
            top = headers[0][1]
            bottom = image.shape[0]
            left = headers[0][0]
            right = headers[-1][0] + headers[-1][2]

            row_height = headers[0][3]
            for y in range(top, bottom, row_height):
                cv2.line(image, (left, y), (right, y), (0, 0, 0), 2)

            cv2.imwrite(local_to_absolute_path('lama 3.1/proverka8/change_imageP') + f'\\result_image_{iterator}.jpg', image)

def main():
    convert_PDF_to_image(pdf_path)
    build_table()
    
# Запуск
pdf_path = Path(r'G:/Python/urfu_LLM_documents/lama 3.1/proverka8/data/pdf_images/Кунашак, КРК, оп 1, п.хр. (3).pdf')

main()
