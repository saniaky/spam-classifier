import os
import numpy as np
import pandas as pd
import mailparser
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from hiddenPrints import HiddenPrints
from processEmail import process_str


def read_emails(emails_dir):
    emails = pd.DataFrame()
    for filename in os.listdir(emails_dir):
        f = open(os.path.join(emails_dir, filename), "r", encoding='ISO-8859-1')
        raw_email = f.read()
        mail = mailparser.parse_from_string(raw_email)
        emails = emails.append({'body': mail.body, 'headers': 'mail.headers'}, ignore_index=True)
    return emails


print("==> Reading emails (wait, it will take a while)...", end=' ')
with HiddenPrints():  # Suppress errors from "mailparser" library
    ham_emails = read_emails('./data/ham')
    spam_emails = read_emails('./data/spam')
ham_emails['spam'] = 0
spam_emails['spam'] = 1
print("Done")

print("==> Merge 2 ham and spam emails...", end=' ')
all_emails = pd.concat([ham_emails, spam_emails])
all_emails.reset_index(drop=True, inplace=True)
del ham_emails, spam_emails  # clear memory
print("Done")

# TfidfVectorizer = CountVectorizer + TfidfTransformer
#  override the built in preprocessor and tokenizer because we already did it
print("==> Building vocabulary, prepossessing, tokenizing sentences...")
vectorizer = TfidfVectorizer(preprocessor=process_str,
                             tokenizer=lambda x: x,  # we already tokenized it in process_str
                             min_df=3)
X = vectorizer.fit_transform(all_emails['body'])
# print(vectorizer.get_feature_names()[0:20])
X = X.toarray()
print("Done")

print("==> Splitting data into train/dev/test datasets...", end=" ")
X_train, X_test, y_train, y_test = train_test_split(
    X, np.array(all_emails['spam']), test_size=0.3, random_state=0)
print("Done")

print("==> Quick test of classifiers with default params...")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("[GaussianNB] TEST score: %.3f" % f1_score(y_test, gnb.predict(X_test)))

clf = svm.LinearSVC()
clf.fit(X_train, y_train)
print("[LinearSVC, C=1] TEST score: %.3f" % f1_score(y_test, clf.predict(X_test)))
print("Done")

print("==> Grid Search hyper-parameters (wait, it will take about 3-5 minutes)...")
pca = PCA(n_components=300)
svm = svm.LinearSVC(C=10, max_iter=2000)
clf.fit(X_train, y_train)

print("==> Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("==> Best parameters set found on development set:")
print(clf.best_params_)
print()

print("==> F-1 score for the TEST set with the best model: %.3f" % f1_score(y_test, clf.predict(X_test)))
