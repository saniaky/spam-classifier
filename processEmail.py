import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

porterStemmer = PorterStemmer()

alphanum = re.compile('[^a-zA-Z0-9]')

# Returns list of words represented as an indexes in vocabulary
def process_email(email):
    email = lowercase(email)
    email = strip_html(email)
    email = normalize_urls(email)
    email = normalize_emails(email)
    email = normalize_numbers(email)
    email = normalize_currency(email)

    words = tokenize(email)
    words = remove_non_words(words)
    words = stemming(words)

    return words


def remove_email_headers(email):
    return ''


# Lower-casing - ignore capitalization
def lowercase(email):
    return email.lower()


# Stripping HTML to leave only content
def strip_html(email):
    return re.sub('<[^<>]+>', ' ', email)


# Normalizing URLs
# All URLs are replaced with placeholder text.
def normalize_urls(email):
    return re.sub('(http|https)://[^\s]*', 'web_addr', email)


# Normalizing Email Addresses
# All email addresses are replaced with placeholder text
def normalize_emails(email):
    return re.sub('[^\s]+@[^\s]+', 'email_addr', email)


# Normalizing Numbers
# All numbers are replaced with the text “number”.
def normalize_numbers(email):
    return re.sub('[0-9]+', 'number', email)


# Normalizing currency symbols
# Foe ex all dollar signs ($) are replaced with placeholder.
def normalize_currency(email):
    return re.sub('[$]+', 'currency', email)


def tokenize(email):
    return re.split('[ @$/#.-:&*+=\[\]?!(){},\'">_<;%\\n\\t]', email)


# Word Stemming
# remove morphological affixes from words, leaving only the word stem.
def stemming(words):
    new_words = [None] * len(words)
    for idx, word in words:
        new_words[idx] = porterStemmer.stem(word)
    return new_words


# Removal of non-words:
def remove_non_words(words):
    # return list(filter(alphanum.search, words))
    return [word for word in words if alphanum.match(word)]
