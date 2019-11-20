import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier

df = pd.read_csv('train.csv')

# choose the data that score is nan as test set
X_test_set = df[df.isnull().Score]

# replace all the nan in text as empty string
test_set = X_test_set['Text'].replace(np.nan, '', regex=True)

# drop nan for training
df.dropna(inplace=True)
print(test_set)


X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Score'], random_state=0)

# print training set size
print('\n\nX_train shape: ', X_train.shape)

# count words in the review text
# vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
vect = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)

# same logistic regression we did in previous hws
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# predict test set results into array
predict_result = model.predict(vect.transform(test_set))
# print(predict_result)
test = pd.read_csv('test.csv')
# print(test['Score'])

# add the results to the id given in test csv and output as result csv
test['Score'] = predict_result
test.to_csv('lr_result.csv', index=False)

# svm method
model2 = LinearSVC()
model2.fit(X_train_vectorized, y_train)

svm_predict = model2.predict(vect.transform(test_set))
test['Score'] = svm_predict
test.to_csv('svm_result.csv', index=False)

# naive bayes method
model3 = MultinomialNB()
model3.fit(X_train_vectorized, y_train)

nb_predict = model3.predict(vect.transform(test_set))
test['Score'] = nb_predict
test.to_csv('nb_result.csv', index=False)

# combined
model4 = VotingClassifier([('lr', model), ('svm', model2), ('nb', model3)])
model4.fit(X_train_vectorized, y_train)

combined_predict = model4.predict(vect.transform(test_set))
test['Score'] = combined_predict
test.to_csv('combined_result.csv', index=False)

# from sklearn.metrics import roc_auc_score
# predictions = model.predict(vect.transform(X_test))
# print('AUC: ', roc_auc_score(y_test, predictions))

# feature_names = np.array(vect.get_feature_names())
# sorted_coef_index = model.coef_[0].argsort()
# print('Smallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
# print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))


# print(model.predict(vect.transform(['Just so so, not good and not bad'])))
