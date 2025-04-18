import os
import csv
import pdfplumber
from langchain_ollama.chat_models import ChatOllama

class PDFChecker:
    def __init__(self):
        self.model_name = "llama3.2"

    def extract_text_with_tables(self, pdf_path):
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            full_text += " | ".join(str(cell).strip() if cell else "" for cell in row) + "\n"

        return full_text

    def analyze(self, text, prompt):
        llm = ChatOllama(model=self.model_name)

        return llm.invoke(f"{prompt}\n\nВот содержимое таблиц из PDF:\n{text}")

    def analyze_dir(self, dir_path, prompt_path, output_dir_path=None):
        if output_dir_path is None:
            output_dir_path = dir_path

        os.makedirs(output_dir_path, exist_ok=True)

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.abspath(os.path.join(dir_path, filename))
                print(f"\nОбработка {pdf_path}")

                try:
                    text = self.extract_text_with_tables(pdf_path)
                    if not text.strip():
                        print(f"✗ Таблицы не найдены в {filename}")
                        continue

                    result = self.analyze(text, prompt)
                    result_text = result["result"] if isinstance(result, dict) and "result" in result else str(result)

                    print(f"\n✓ Результат:\n\n{result_text}")

                    # Если это таблица в markdown — сохраняем как .csv
                    if "|" in result_text and result_text.count("\n") > 3:
                        lines = [line.strip() for line in result_text.split("\n") if line.strip().startswith("|")]
                        if len(lines) >= 3:
                            headers = [h.strip() for h in lines[0].strip("|").split("|")]
                            data = [
                                [cell.strip() for cell in row.strip("|").split("|")]
                                for row in lines[2:]
                            ]
                            csv_filename = os.path.splitext(filename)[0] + ".csv"
                            output_csv_path = os.path.join(output_dir_path, csv_filename)
                            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow(headers)
                                writer.writerows(data)

                            print(f"✓ CSV сохранён в {output_csv_path}")
                except Exception as e:
                    print(f"\n✗ Ошибка при обработке {filename}: {e}")

def main():
    checker = PDFChecker()
    checker.analyze_dir(dir_path="../docs/22", prompt_path="./prompt.txt")

if __name__ == "__main__":
    main()
