import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# https://pypi.org/project/mail-parser/
import mailparser
from processEmail import process_str


def read_emails(emails_dir):
    emails = pd.DataFrame()
    for filename in os.listdir(emails_dir):
        f = open(os.path.join(emails_dir, filename), "r", encoding='ISO-8859-1')
        raw_email = f.read()
        mail = mailparser.parse_from_string(raw_email)
        emails = emails.append({'body': mail.body, 'headers': 'mail.headers'}, ignore_index=True)
    return emails


ham_emails = read_emails('./tmp/ham')
spam_emails = read_emails('./tmp/spam')
ham_emails['spam'] = 0
spam_emails['spam'] = 1

print("==> Merge 2 ham and spam emails...")
all_emails = pd.concat([ham_emails, spam_emails])
all_emails.reset_index(drop=True, inplace=True)
# print(all_emails)
print("Done")

# pre-process emails
print("==> Pre-processing emails...")
for index, email in all_emails.iterrows():
    if len(email['body']) > 0:
        processed_email = process_str(email['body'])
print("Done.")

# CountVectorizer + TfidfTransformer
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode')
X = vectorizer.fit_transform(all_emails['body'])
print(vectorizer.get_feature_names())
print(X.toarray())

print("==> Reducing data dimension...", end=' ')
pca = PCA(n_components=200)
X_train_reduced = pca.fit_transform(X)
print("Done")

print("==> Splitting data into train/dev/test datasets...", end=" ")
# print("==> Vectorized version of it:\n%s\n" % vectorizer.transform([sample]))
# X_train, X_test, y_train, y_test = train_test_split(
#     all_data_vectorized, np.array(all_data['y']), test_size=0.2, random_state=1)
print("Done")


print("==> Training an SVM classifier using k-fold cross validation...")
#clf = svm.LinearSVC(C=1)
#clf.fit(X)
#scores = k_fold_cv(clf, X_train_reduced, y_train)
#print("==> Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), end=' ')
print("Done")

print("==> Grid Search...")

