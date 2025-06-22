import json
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from Levenshtein import ratio

# Фильтрация наименований
def filter_names(names):
    # Регулярное выражение для удаления слов и спецсимволов
    pattern = r'\b(УТВЕРЖДАЮ|УТВЕРЖДЕНО|СОГЛАСОВАНО|Протокол|ЭК|г|от|год)\b|\d+|[^\w\s]'
    filtered_names = []

    for name in names:
        # Приводим к нижнему регистру наименования
        name = name.lower()
        # Применяем регулярное выражение для удаления
        filtered_name = re.sub(pattern, '', name, flags=re.IGNORECASE).strip()
        filtered_names.append(filtered_name)
    return filtered_names

# Загружаем наименования из файла
def load_names_from_file():
    with open('proverka8/data/BlockNames.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["approve_arr"], data["agreed_arr"]

# Преобразуем наименования в эмбеддинги
def create_embeddings(names):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model.encode(names)

# Применяем FAISS для сравнения
def compare_with_faiss(approve_arr, agreed_arr):
    embeddings_approve = create_embeddings(approve_arr)
    embeddings_agreed = create_embeddings(agreed_arr)

    # Размерность векторов
    dim = embeddings_approve.shape[1]
    index = faiss.IndexFlatL2(dim)

    # Добавляем эмбеддинги УТВЕРЖДАЮ в индекс
    index.add(np.array(embeddings_approve).astype('float32'))

    # Ищем ближайший сосед
    D, I = index.search(np.array(embeddings_agreed).astype('float32'), k=1)

    # for i in range(len(D)):
    #     print(f"Схожесть между наименованиями УТВЕРЖДАЮ и СОГЛАСОВАНО: {D[i][0]}")
    #     if D[i][0] < 0.5:  # Если схожесть выше порога (например, 0.5)
    #         print("Наименования схожи.")
    #     else:
    #         print("Наименования не схожи.")

# Основная логика
def main():
    approve_arr, agreed_arr = load_names_from_file()

    # Применяем фильтрацию наименований
    approve_arr = filter_names(approve_arr)
    agreed_arr = filter_names(agreed_arr)

    # Пропечатываем отфильтрованные от лишних символов и слов наименования
    for item in approve_arr:
        print("\n Отфильтрованные наименования:", "\n" + item)
    for item in agreed_arr:
        print(item)

    # Проводим сверку
    if approve_arr and agreed_arr:
        if len(approve_arr) == 1 and len(agreed_arr) == 1:
            # Сравнение для одного наименования
            similarity = ratio(approve_arr[0], agreed_arr[0])
            if similarity <= 0.6:
                print(f"\nВНИМАНИЕ!\nНизкий процент совпадения текста (схожесть: {similarity:.2f})")
            else:
                print(f"\nНаименования совпадают (схожесть: {similarity:.2f})")
        else:
            # Если наименований больше одного, сравниваем каждую пару
            if len(approve_arr) > 1:
                print("Обнаружено больше одного наименования 'УТВЕРЖДАЮ'. Сравниваем:")
                for i in range(len(approve_arr)):
                    for j in range(len(agreed_arr)):
                        similarity = ratio(approve_arr[i], agreed_arr[j])
                        if similarity <= 0.6:
                            print(f"Наименования не совпадают (различия: {similarity:.2f})")
                        else:
                            print(f"Наименования совпадают (схожесть: {similarity:.2f})")

            if len(agreed_arr) > 1:
                print("Обнаружено больше одного наименования 'СОГЛАСОВАНО'. Сравниваем:")
                for i in range(len(approve_arr)):
                    for j in range(len(agreed_arr)):
                        similarity = ratio(approve_arr[i], agreed_arr[j])
                        if similarity <= 0.6:
                            print(f"Наименования не совпадают (различия: {similarity:.2f})")
                        else:
                            print(f"Наименования совпадают (схожесть: {similarity:.2f})")
    else:
        print("Ошибка! Наименования не найдены!")

    # Выполняем сравнение с использованием FAISS
    compare_with_faiss(approve_arr, agreed_arr)

if __name__ == "__main__":
    main()
