from processEmail import process_email
from vocabulary import map_words

# Pre-process email
words = process_email("<html>HELLO world pythoner! It's $100, how are \n you're \t doing? http://ya.ru</html>")

# For each word in the email find it's index in the vocabulary
words_indices = map_words(words)

# Train classifier
# todo

# Predict
# todo

# Analyze
# todo
