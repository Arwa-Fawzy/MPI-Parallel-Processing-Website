import csv
import random

def generate_csv(filename='random_data.csv', rows=100):
    # Define columns and their value ranges (min, max)
    columns = {
        'Age': (18, 70),
        'Height_cm': (150, 200),
        'Weight_kg': (50, 120),
        'Income_k': (20, 150),
        'Score': (0, 100)
    }

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(columns.keys())

        for _ in range(rows):
            row = []
            for col_range in columns.values():
                val = round(random.uniform(*col_range), 2)
                row.append(val)
            writer.writerow(row)

    print(f"CSV file '{filename}' generated with {rows} rows.")

if __name__ == "__main__":
    generate_csv()
