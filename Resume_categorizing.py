import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv' ,encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''
resumeDataSet.head()
resumeDataSet.info()
print ("Displaying the distinct categories of resume:\n\n ")
print (resumeDataSet['Category'].unique())
print ("Displaying the distinct categories of resume and the number of records belonging to each category:\n\n")
print (resumeDataSet['Category'].value_counts())
import seaborn as sns
plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
ax=sns.countplot(x="Category", data=resumeDataSet)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.grid()
from matplotlib.gridspec import GridSpec
targetCounts = resumeDataSet['Category'].value_counts()
targetLabels  = resumeDataSet['Category'].unique()
# Make square figures and axes
plt.figure(1, figsize=(22,22))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('coolwarm')
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True)
plt.show()
import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
resumeDataSet.head()
resumeDataSet_d=resumeDataSet.copy()
from sklearn.preprocessing import LabelEncoder

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
resumeDataSet.head()
resumeDataSet.Category.value_counts()
resumeDataSet_d.Category.value_counts() #understanding decode LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print ("Feature completed .....")

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,
                                                 shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
clf = OneVsRestClassifier(GaussianNB())
clf.fit(X_train.toarray(), y_train)
prediction = clf.predict(X_test.toarray())
print('Accuracy of Naive Bayes classifiers on training set: {:.2f}'.format(clf.score(X_train.toarray(), y_train)))
print('Accuracy of Naive Bayes classifiers on test set:     {:.2f}'.format(clf.score(X_test.toarray(), y_test)))
from sklearn.tree import DecisionTreeClassifier
clf = OneVsRestClassifier(DecisionTreeClassifier())
clf.fit(X_train.toarray(), y_train)
prediction = clf.predict(X_test.toarray())
print('Accuracy of Decision Tree Classifier on training set: {:.2f}'.format(clf.score(X_train.toarray(), y_train)))
print('Accuracy of Decision Tree Classifier on test set:     {:.2f}'.format(clf.score(X_test.toarray(), y_test)))
from sklearn.ensemble import RandomForestClassifier
clf = OneVsRestClassifier(RandomForestClassifier())
clf.fit(X_train.toarray(), y_train)
prediction = clf.predict(X_test.toarray())
print('Accuracy of Random Forest Classifier on training set: {:.2f}'.format(clf.score(X_train.toarray(), y_train)))
print('Accuracy of Random Forest Classifier on test set:     {:.2f}'.format(clf.score(X_test.toarray(), y_test)))
from sklearn.linear_model import LogisticRegression
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train.toarray(), y_train)
prediction = clf.predict(X_test.toarray())
print('Accuracy of Logistic regression on training set: {:.2f}'.format(clf.score(X_train.toarray(), y_train)))
print('Accuracy of Logistic regression on test set:     {:.2f}'.format(clf.score(X_test.toarray(), y_test)))
from sklearn.svm import SVC
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of Support Vector Machine on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Support Vector Machine on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of AdaBoost Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of AdaBoost Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
clf = MLPClassifier(alpha=1, max_iter=1000)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
