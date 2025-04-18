import os
import csv
import fitz  # PyMuPDF
from langchain_ollama.chat_models import ChatOllama

class PDFChecker:
    def __init__(self):
        self.model_name = "llama3.2"

    def extract_text_with_fitz(self, pdf_path):
        full_text = ""
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            sorted_blocks = sorted(blocks, key=lambda b: (round(b[1]), round(b[0])))
            for block in sorted_blocks:
                text = block[4].strip()
                if text:
                    full_text += text + "\n"
        return full_text

    def analyze(self, text, prompt):
        llm = ChatOllama(model=self.model_name)
        response = llm.invoke(f"{prompt}\n\nВот содержимое таблиц из PDF:\n{text}")
        return response["message"]["content"] if isinstance(response, dict) else str(response)

    def save_clean_csv(self, text, output_path):
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        csv_lines = []

        for line in lines:
            try:
                parsed = next(csv.reader([line], skipinitialspace=True))
                if 3 <= len(parsed) <= 6:
                    csv_lines.append(parsed)
            except Exception:
                continue

        if not csv_lines:
            print("✗ Не найдено валидных CSV-строк")
            return

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_lines)

        print(f"✓ CSV сохранён в {output_path}")

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
                    text = self.extract_text_with_fitz(pdf_path)
                    if not text.strip():
                        print(f"✗ Не удалось извлечь текст из {filename}")
                        continue

                    result_text = self.analyze(text, prompt)
                    print(f"\n✓ Результат:\n\n{result_text}")

                    output_csv_path = os.path.join(output_dir_path, os.path.splitext(filename)[0] + ".csv")
                    self.save_clean_csv(result_text, output_csv_path)

                except Exception as e:
                    print(f"\n✗ Ошибка при обработке {filename}: {e}")

def main():
    checker = PDFChecker()
    checker.analyze_dir(dir_path="../docs/22", prompt_path="./prompt.txt")

if __name__ == "__main__":
    main()