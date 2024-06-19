import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import dice_ml

#create dataframe 
dataframe = pd.read_csv('Tabella_corsi.csv')
print(dataframe)

#preprocessing dataframe
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

#Logistic regression
decisiontreeClssifier = DecisionTreeClassifier()
decisiontreeClssifier.fit(X_train, y_train)
predictions = decisiontreeClssifier.predict(X_test) #predictions on test data
print("superamento corso")
print(predictions)

#classification report
report = classification_report(y_test,predictions)
print(report)

#confusion matrix 
cm = confusion_matrix(y_test, predictions)
print("confusion matrix")
print(cm)

#accuracy
accuracy = accuracy_score(y_test, predictions)
print("accuracy")
print(accuracy)

#inizialize DiCE
diceData = dice_ml.Data(dataframe=dataframe,continuous_features=[], outcome_name="Superamento_Corso")
diceModel = dice_ml.Model(model=decisiontreeClssifier, backend="sklearn")
exp = dice_ml.Dice(diceData, diceModel, method='kdtree')

#division into samples
positive_samples = dataframe[dataframe["Superamento_Corso"] == 1].drop_duplicates("Id_Corso").sample(4)
negative_samples = dataframe[dataframe["Superamento_Corso"] == 0].drop_duplicates("Id_Corso").sample(4)

#generate counterfactual Explanation
query_instance = pd.concat([positive_samples, negative_samples]).drop(columns="Superamento_Corso")
counterFactual = exp.generate_counterfactuals(query_instances=query_instance, total_CFs=3,desired_class="opposite")
counterFactual.visualize_as_dataframe()

#global importance
global_importance = exp.global_feature_importance(X_train[0:10], total_CFs=10, posthoc_sparsity_param=None, desired_class="opposite")
print("Global importance",global_importance.summary_importance)