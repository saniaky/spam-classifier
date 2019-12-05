import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

porterStemmer = PorterStemmer()

alphanum = re.compile('[^a-zA-Z0-9]+')


# Returns list of words represented as an indexes in vocabulary
def process_str(sample):
    if sample is None: return []

    sample = lowercase(sample)
    sample = strip_html(sample)
    sample = normalize_urls(sample)
    sample = normalize_emails(sample)
    sample = normalize_numbers(sample)
    sample = normalize_currency(sample)

    words = tokenize(sample)
    words = remove_non_words(words)
    words = remove_empty_strings(words)
    words = stemming(words)

    return words


# Lower-casing - ignore capitalization
def lowercase(sample):
    return sample.lower()


# Stripping HTML to leave only content
def strip_html(email):
    return re.sub('<[^<>]+>', ' ', email)


# Normalizing URLs
# All URLs are replaced with placeholder text.
def normalize_urls(email):
    return re.sub('(http|https)://[^\s]*', 'web_addr', email)


# Normalizing Email Addresses
# All email addresses are replaced with placeholder text
def normalize_emails(sample):
    return re.sub('[^\s]+@[^\s]+', 'email_addr', sample)


# Normalizing Numbers
# All numbers are replaced with the text “number”.
def normalize_numbers(sample):
    return re.sub('[0-9]+', 'number', sample)


# Normalizing currency symbols
# Foe ex all dollar signs ($) are replaced with placeholder.
def normalize_currency(sample):
    return re.sub('[$]+', 'currency', sample)


def tokenize(sample):
    return re.split('[ @$/#.-:&*+=\[\]?!(){},\'">_<;%\\n\\t]', sample)


# Removal of non-words:
def remove_non_words(words):
    return [word for word in words if not alphanum.match(word)]


def remove_empty_strings(words):
    return [i for i in words if i]

# Word Stemming
# remove morphological affixes from words, leaving only the word stem.
def stemming(words):
    new_words = [None] * len(words)
    for idx, word in enumerate(words):
        new_words[idx] = porterStemmer.stem(word)
    return new_words
