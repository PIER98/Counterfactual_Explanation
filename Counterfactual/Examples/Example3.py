import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import dice_ml
#create dataframe
dataframe = pd.read_csv('Tabella_Corsi.csv')
print(dataframe)

#preprocessing datafreame
labelEncoder = LabelEncoder()
dataframe['Superamento_Corso'] = labelEncoder.fit_transform(dataframe['Superamento_Corso']) #{ Non_superato:0, Superato: 1}
print(dataframe.head())

#drop useless columns
dataframe = dataframe.drop('VOTO', axis=1)
dataframe = dataframe.drop('Subject Id', axis=1)

#create target 
X = dataframe.drop(columns="Superamento_Corso")
y = dataframe['Superamento_Corso']

#split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#let's see how the dataset was splitted
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)

#K-nearest-neighbors ForestClassifier
KNNClassifier = KNeighborsClassifier(n_neighbors=5)
KNNClassifier.fit(X_train, y_train)
predictions = KNNClassifier.predict(X_test)
print("superamento corso")
print(predictions)

#classification report
report = classification_report(y_test, predictions)
print(report)

#confusion matrix 
cm = confusion_matrix(y_test, predictions)
print("confusion matrix")
print(cm)

#accuracy
accuracy = accuracy_score(y_test, predictions)
print("accuracy")
print(accuracy)