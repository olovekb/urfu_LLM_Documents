import pandas as pd
import ollama

class CSVClassifierWithOllama:
    def __init__(self, csv_path, output_path):
        self.csv_path = csv_path
        self.output_path = output_path
        self.model_name = "llama3.2"

    def build_prompt(self, lines):
        # numbered_lines = "\n".join([f"{i + 1}. {line}" for i, line in enumerate(lines)])
        return (
            "Классифицируй каждую строку ниже как одну из категорий: "
            "'heading' (заголовок), 'table' (таблица), 'text' (текст), 'meta' (служебная информация). "
            "И пришли ответ Всех строк также только к каждой добавь категорию"
            "Строки:"
            f"{lines}"
        )

    def parse_response(self, response, original_lines):
        labels = []
        lines = response.strip().split("\n")
        for i, original in enumerate(original_lines):
            found = False
            for line in lines:
                if line.strip().startswith(f"{i + 1}."):
                    for label in ["heading", "table", "text", "meta"]:
                        if label in line.lower():
                            labels.append(label)
                            found = True
                            break
                    break
            if not found:
                labels.append("unknown")
        return pd.DataFrame({"Строка": original_lines, "Тип": labels})

    def classify_lines(self, lines):
        prompt = self.build_prompt(lines)
        response = ollama.chat(model=self.model_name, messages=[
            {"role": "user", "content": prompt}
        ])
        print(response['message'])
        return self.parse_response(response['message']['content'], lines)

    def run(self):
        df = pd.read_csv(self.csv_path)
        lines = df.iloc[:, 0].dropna().astype(str).tolist()
        print(f"Обработка {len(lines)} строк...")
        result_df = self.classify_lines(lines)
        result_df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"Результат сохранён в {self.output_path}")

def main():
    input_path = "../docs/22/csv/Еткуль, УСЗН, оп 1, п.хр.2021.csv"
    output_path = "../docs/22/csv/classified_output.csv"
    classifier = CSVClassifierWithOllama(input_path, output_path)
    classifier.run()

if __name__ == "__main__":
    main()
