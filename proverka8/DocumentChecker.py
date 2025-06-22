import cv2
import numpy as np
import os
from cv2.gapi import kernel
from pdf2image import convert_from_path
from pathlib import Path

def local_to_absolute_path(file_path):
    return str(Path(file_path).resolve())

# Преобразование pdf к jpg
def convert_PDF_to_image(file_path, save_image_path = local_to_absolute_path('proverka8/data/pdf_images'), poppler_path = local_to_absolute_path('urfu_LLM_documents/Libraries/poppler-24.08.0/Library/bi')): 
    print('Шаг 1 Конвертация pdf к изображению...')
    os.environ["PATH"] += os.pathsep + poppler_path
    images = convert_from_path(file_path)

    for i, image in enumerate(images, start=1):
        image.save(f'{save_image_path}\\page_{i}.jpg', 'JPEG')

# Построение таблицы
def build_table(images_path = local_to_absolute_path('proverka8/data/pdf_images')):

    for iterator in range(1, len(os.listdir(images_path)) + 1):
        image_path = local_to_absolute_path('proverka8/data/pdf_images') + f'\\page_{iterator}.jpg'
        image = cv2.imread(image_path)
        
        # Проверка, что изображение успешно загружено
        if image is None:
            print(f"Не удалось загрузить изображение {image_path}")
            continue  # Переход к следующему изображению, если не удалось загрузить текущее
        
        filterd_image = cv2.medianBlur(image, 1)
        gray = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)

        # Бинаризация
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

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

            cv2.imwrite(local_to_absolute_path('proverka8/change_image') + f'\\result_image_{iterator}.jpg', image)

# Удаление файлов из всех папок
def delete_file_in_folder():
    folders_to_delete = [
        r'G:\Python\urfu_LLM_documents\proverka8\change_image',
        r'G:\Python\urfu_LLM_documents\proverka8\gray_image',
        r'G:\Python\urfu_LLM_documents\proverka8\blur_image',
        r'G:\Python\urfu_LLM_documents\proverka8\threshold_image',
        r'G:\Python\urfu_LLM_documents\proverka8\kernal_image',
        r'G:\Python\urfu_LLM_documents\proverka8\dilate_image',
        r'G:\Python\urfu_LLM_documents\proverka8\bbox_image',
        r'G:\Python\urfu_LLM_documents\proverka8\cropped_image'

    ]

    for folder in folders_to_delete:
        folder_path = local_to_absolute_path(folder) + '\\'
        if os.path.exists(folder_path):  # Проверяем, существует ли папка
            for value in os.listdir(folder_path):
                try:
                    file_path = os.path.join(folder_path, value)
                    if os.path.isfile(file_path) and value.endswith(".jpg"):
                        os.remove(file_path)
                        print(f'Удален файл {file_path}')
                except Exception as e:
                    print(f'Ошибка при удалении файла {file_path}. {e}')
        else:
            print(f"Папка не существует: {folder_path}")
            os.makedirs(folder_path)  # Создаем папку, если она не существует

# Преобразование всех изображений в серые
def convert_to_gray():
    image_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\change_image')
    gray_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\gray_image')
    
    # Проходим по изображениям в папке change_image и конвертируем их в серые
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(image_folder, filename)
            image = cv2.imread(img_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Преобразование в серый цвет
            
            # Сохраняем результат в папке gray_image
            gray_path = os.path.join(gray_folder, filename)
            cv2.imwrite(gray_path, gray_image)
            print(f"Сохранено серое изображение: {gray_path}")

# Применение размытия к изображениям
def apply_gaussian_blur():
    gray_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\gray_image')
    blur_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\blur_image')
    
    # Проходим по изображениям в папке gray_image и применяем размытие
    for filename in os.listdir(gray_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(gray_folder, filename)
            image = cv2.imread(img_path)
            blurred_image = cv2.GaussianBlur(image, (7, 7), 0)  # Применение размытия
            
            # Сохраняем результат в папке blur_image
            blur_path = os.path.join(blur_folder, filename)
            cv2.imwrite(blur_path, blurred_image)
            print(f"Сохранено размытое изображение: {blur_path}")

# Применение пороговой обработки
def apply_threshold():
    blur_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\blur_image')
    threshold_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\threshold_image')
    
    # Проходим по изображениям в папке blur_image и применяем пороговую обработку
    for filename in os.listdir(blur_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(blur_folder, filename)
            image = cv2.imread(img_path)
            
            # Преобразуем изображение в оттенки серого (если оно не серое)
            if len(image.shape) == 3:  # если изображение цветное
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Применяем пороговую обработку
            ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Сохраняем результат в папке threshold_image
            threshold_path = os.path.join(threshold_folder, filename)
            cv2.imwrite(threshold_path, thresh)
            print(f"Сохранено пороговое изображение: {threshold_path}")

# Применение техники морфологического преобразования (kernal)
def apply_kernal():
    threshold_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\threshold_image')
    kernal_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\kernal_image')
    
    # Создаем ядро для морфологической операции
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    
    # Проходим по изображениям в папке threshold_image и применяем морфологическое преобразование
    for filename in os.listdir(threshold_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(threshold_folder, filename)
            image = cv2.imread(img_path)
            
            # Сохраняем результат в папке kernal_image
            kernal_path = os.path.join(kernal_folder, filename)
            cv2.imwrite(kernal_path, kernal)
            print(f"Сохранено изображение с морфологическим преобразованием: {kernal_path}")

# Применение техники dilate
def apply_dilate():
    threshold_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\threshold_image')
    dilate_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\dilate_image')
    
    # Создаем ядро для dilate
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 27))
    
    # Проходим по изображениям в папке threshold_image и применяем dilate
    for filename in os.listdir(threshold_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(threshold_folder, filename)
            image = cv2.imread(img_path)
            
            # Применяем dilate
            dilate = cv2.dilate(image, kernal, iterations=1)
            
            # Сохраняем результат в папке dilate_image
            dilate_path = os.path.join(dilate_folder, filename)
            cv2.imwrite(dilate_path, dilate)
            print(f"Сохранено изображение с dilate: {dilate_path}")

def build_table_after_preprocessing(images_path = local_to_absolute_path('proverka8/dilate_image')):
    bbox_image_folder = r'G:\Python\urfu_LLM_documents\proverka8\bbox_image'
    change_image_folder = r'G:\Python\urfu_LLM_documents\proverka8\change_image'  
    сropped_image_folder = r'G:\Python\urfu_LLM_documents\proverka8\cropped_image'  


    for iterator in range(1, len(os.listdir(images_path)) + 1):
        dilate_path = os.path.join(images_path, f'result_image_{iterator}.jpg')
        dilate = cv2.imread(dilate_path)

        # Проверка, что изображение прошло dilate и успешно загружено
        if dilate is None:
            print(f"Не удалось загрузить изображение {dilate_path}")
            continue

        image_path = os.path.join(change_image_folder, f'result_image_{iterator}.jpg')
        image = cv2.imread(image_path)

        # Проверка, что исходное изображение успешно загружено
        if image is None:
            print(f"Не удалось загрузить исходное изображение {image_path}")
            continue

        # Преобразуем изображение dilate в серый цвет для поиска контуров
        gray_dilate = cv2.cvtColor(dilate, cv2.COLOR_BGR2GRAY)

        # Применение пороговой обработки для выделения контуров
        _, thresh = cv2.threshold(gray_dilate, 150, 255, cv2.THRESH_BINARY_INV)

        # Морфологическое расширение для улучшения контуров
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 30))  # Увеличиваем размер ядра
        dilated = cv2.dilate(thresh, kernal, iterations=1)

        # Поиск контуров на изображении
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Сортировка контуров по координате X (влево-направо)
        cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

        # Рисование прямоугольников для каждого контура на исходном изображении
        cropped_image_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\proverka8\cropped_image')

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 150 and h > 250:  # Фильтрация по размерам столбцов
                roi = image[y:y+h, x:x+h]
                cropped_image_path = os.path.join(cropped_image_folder, f"cropped_image_{iterator}.jpg")
                cv2.imwrite(cropped_image_path, roi)
                print(f"Сохранено обрезанное изображение: {cropped_image_path}")
                cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            
        # Сохранение изображения с прямоугольниками
        result_image_path = os.path.join(bbox_image_folder, f'result_image_{iterator}.jpg')
        cv2.imwrite(result_image_path, image)
        print(f"Сохранено изображение с контурами: {result_image_path}")


def main():
    delete_file_in_folder()  # Удаляем файлы из всех папок
    pdf_path = r'G:\Python\urfu_LLM_documents\proverka8\data\pdf_images\Кунашак, КРК, оп 1, п.хр. (3).pdf'  # Укажите путь к своему PDF
    convert_PDF_to_image(pdf_path)
    build_table()
    convert_to_gray()  # Преобразуем все изображения в серые
    apply_gaussian_blur()  # Применяем размытие к изображениям
    apply_threshold()  # Применяем пороговую обработку к изображениям
    apply_kernal()  # Применяем морфологическое преобразование (kernal)
    apply_dilate()  # Применяем dilate к изображениям
    build_table_after_preprocessing()

main()