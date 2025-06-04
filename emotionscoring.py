

#Importing required libraries for Python Essentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
#Importing required libraries for scikit learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris



df = pd.read_csv('corpus.csv')



X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#Random forest implementation with
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)



feature_importances = classifier.feature_importances_

plt.barh(x=['fear', 'hate','anger','empathy','pride'], y=feature_importances)
plt.xlabel('Emotion Score')
plt.title('Emotion Score in Random Forest Classifier')
plt.show()
