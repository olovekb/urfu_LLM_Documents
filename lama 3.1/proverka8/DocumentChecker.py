import os
import cv2
import fitz  # PyMuPDF
from pathlib import Path

# Функция для преобразования относительных путей в абсолютные
def local_to_absolute_path(file_path):
    return str(Path(file_path).resolve())

# Преобразование pdf в jpg с использованием PyMuPDF
def convert_PDF_to_image(file_path, save_image_path = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\change_image')):
    print('Шаг 1: Конвертация pdf в изображение...')
    
    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        print(f"Ошибка: файл {file_path} не найден.")
        return
    
    # Открываем PDF файл с использованием PyMuPDF
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Ошибка при открытии PDF: {e}")
        return
    
    # Проходим по всем страницам PDF
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Загружаем страницу
        pix = page.get_pixmap()  # Получаем изображение страницы
        output_path = os.path.join(save_image_path, f"page_{page_num + 1}.jpg")
        pix.save(output_path)  # Сохраняем изображение в формате JPG
        print(f"Страница {page_num + 1} сохранена как {output_path}")

# Удаление файлов из всех папок
def delete_file_in_folder():
    folders_to_delete = [
        r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\change_image',
        r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\gray_image',
        r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\blur_image',
        r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\threshold_image',
        r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\kernal_image',
        r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\dilate_image'
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
    image_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\change_image')
    gray_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\gray_image')
    
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
    gray_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\gray_image')
    blur_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\blur_image')
    
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
    blur_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\blur_image')
    threshold_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\threshold_image')
    
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
    threshold_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\threshold_image')
    kernal_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\kernal_image')
    
    # Создаем ядро для морфологической операции
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    
    # Проходим по изображениям в папке threshold_image и применяем морфологическое преобразование
    for filename in os.listdir(threshold_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(threshold_folder, filename)
            image = cv2.imread(img_path)
            
            # Применяем морфологическое преобразование (dilation)
            kernal_image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernal)
            
            # Сохраняем результат в папке kernal_image
            kernal_path = os.path.join(kernal_folder, filename)
            cv2.imwrite(kernal_path, kernal_image)
            print(f"Сохранено изображение с морфологическим преобразованием: {kernal_path}")

# Применение техники dilate
def apply_dilate():
    threshold_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\threshold_image')
    dilate_folder = local_to_absolute_path(r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\dilate_image')
    
    # Создаем ядро для dilate
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    
    # Проходим по изображениям в папке threshold_image и применяем dilate
    for filename in os.listdir(threshold_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(threshold_folder, filename)
            image = cv2.imread(img_path)
            
            # Применяем dilate
            dilated_image = cv2.dilate(image, kernal, iterations=1)
            
            # Сохраняем результат в папке dilate_image
            dilate_path = os.path.join(dilate_folder, filename)
            cv2.imwrite(dilate_path, dilated_image)
            print(f"Сохранено изображение с dilate: {dilate_path}")

def main():
    delete_file_in_folder()  # Удаляем файлы из всех папок
    pdf_path = r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\data\pdf_images\Кунашак, КРК, оп 1, п.хр. (3).pdf'  # Укажите путь к своему PDF
    convert_PDF_to_image(pdf_path)  # Конвертируем PDF в изображения
    convert_to_gray()  # Преобразуем все изображения в серые
    apply_gaussian_blur()  # Применяем размытие к изображениям
    apply_threshold()  # Применяем пороговую обработку к изображениям
    apply_kernal()  # Применяем морфологическое преобразование (kernal)
    apply_dilate()  # Применяем dilate к изображениям

# Запуск
main()
