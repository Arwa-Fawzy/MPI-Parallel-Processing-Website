import random

def generate_text_file(filename="random_words.txt", total_words=1000):
    # A small sample list of English words
    sample_words = [
        "apple", "banana", "orange", "grape", "melon",
        "kiwi", "pear", "peach", "plum", "cherry",
        "table", "chair", "sofa", "desk", "lamp",
        "car", "bike", "train", "plane", "boat",
        "cat", "dog", "mouse", "rabbit", "horse",
        "red", "blue", "green", "yellow", "purple",
        "run", "walk", "jump", "sit", "stand",
        "happy", "sad", "angry", "excited", "bored",
        "sun", "moon", "star", "cloud", "rain",
        "city", "village", "country", "forest", "mountain"
    ]

    words = []
    while len(words) < total_words:
        word = random.choice(sample_words)
        # Randomly decide to add repeats of this word 1 to 3 times
        repeats = random.choices([1,2,3], weights=[0.7,0.2,0.1])[0]
        words.extend([word]*repeats)

    # Trim excess if any
    words = words[:total_words]

    # Join words with spaces and write to file
    with open(filename, 'w') as f:
        f.write(" ".join(words))

    print(f"Generated file '{filename}' with {len(words)} words.")

if __name__ == "__main__":
    generate_text_file()
