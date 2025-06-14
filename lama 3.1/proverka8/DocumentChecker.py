import cv2
import pytesseract
import os
from pdf2image import convert_from_path
from pathlib import Path

# Укажите путь к tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'G:\Python\urfu_LLM_documents\lama 3.1\Tesseract\tesseract.exe'

# Функция для преобразования PDF в JPEG с указанием пути к Poppler
def convert_pdf_to_jpeg(pdf_path, output_folder, poppler_path):
    # Преобразуем пути в абсолютные пути для корректной работы с кириллицей
    pdf_path = Path(pdf_path).resolve()
    output_folder = Path(output_folder).resolve()
    
    if not pdf_path.exists():
        print(f"Файл не найден: {pdf_path}")
        return

    os.environ["PATH"] += os.pathsep + poppler_path  # Добавляем Poppler в PATH
    try:
        images = convert_from_path(str(pdf_path))  # Преобразование PDF в изображения
        for i, image in enumerate(images, start=1):
            image.save(str(output_folder / f'page_{i}.jpg'), 'JPEG')
    except Exception as e:
        print(f"Ошибка при преобразовании PDF: {e}")

# Путь к папке для сохранения изображений
image_folder = r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\change_image'
pdf_path = r'G:\Python\urfu_LLM_documents\lama 3.1\proverka8\data\МАГНИТ_МКУ,_ЗАО_БЕЛКА,_Ф_Р_507,_ОП_697,ЛС_1.pdf'
poppler_path = r'G:\Python\urfu_LLM_documents\lama 3.1\poppler-24.08.0\Library\bin'

# Преобразование PDF в JPEG
convert_pdf_to_jpeg(pdf_path, image_folder, poppler_path)

# Получаем список всех файлов в папке
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Обрабатываем каждое изображение
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применение бинаризации для выделения контуров
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по размеру (можно настроить в зависимости от таблицы)
    min_width, min_height = 50, 20
    filtered_contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > min_width and cv2.boundingRect(cnt)[3] > min_height]

    # Перебор контуров для нахождения текста в таблице
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y+h, x:x+w]

        # Извлечение текста с помощью pytesseract
        text = pytesseract.image_to_string(roi, lang='rus', config='--psm 6')

        # Проверка на ключевые слова для столбцов
        keywords = ['№ п/п', 'Индекс дела', 'Заголовок дела', 'Крайние даты', 'Кол-во листов', 'Примечание']
        if any(keyword in text for keyword in keywords):
            # Рисуем рамку вокруг найденного текста
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 255, 12), 2)
            print(f"Текст найден в области {x}, {y}, {w}, {h}: {text}")

    # Сохраняем изображение с рамками
    cv2.imwrite(f'{os.path.join(image_folder, "annotated_" + image_file)}', img)
