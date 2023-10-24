import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns

loans = pd.read_csv("loan_data.csv")
print(loans.info())
print(loans.describe())
print(loans.head())

plt.figure(figsize=(10,6))
loans[loans["credit.policy"] == 1]["fico"].hist(bins = 35, color = "blue",
                                                label = "Credit Policy = 1",
                                                alpha = 0.6)
loans[loans["credit.policy"] == 0]["fico"].hist(bins = 35, color = "red",
                                                label = "Credit Policy = 0",
                                                alpha = 0.6)
plt.legend()
plt.xlabel("FICO")
plt.savefig("credit_policy")
plt.close()


plt.figure(figsize=(10,6))
loans[loans["not.fully.paid"] == 1]["fico"].hist(bins = 35, color = "blue",
                                                label = "not.fully.paid = 1",
                                                alpha = 0.6)
loans[loans["not.fully.paid"] == 0]["fico"].hist(bins = 35, color = "red",
                                                label = "not.fully.paid = 0",
                                                alpha = 0.6)
plt.legend()
plt.xlabel("FICO")
plt.savefig("fully_paid")
plt.close()


plt.figure(figsize=(11,7))
sns.countplot(x="purpose", hue = "not.fully.paid", data = loans, palette = "Set1")
plt.savefig("purpose")
plt.close()

sns.jointplot(x="fico", y = "int.rate", data=loans, color = "purple")
plt.savefig("joinplot")
plt.close()

plt.figure(figsize=(11,7))
sns.lmplot(y="int.rate", x = "fico", data=loans, hue = "credit.policy",
           col = "not.fully.paid", palette = "Set1")
plt.savefig("lmplot")
plt.close()

cat_feats = ["purpose"]
final_data = pd.get_dummies(loans, columns = cat_feats, drop_first=True)

print(final_data.head())

from sklearn.model_selection import train_test_split

X = final_data.drop("not.fully.paid", axis = 1)
y = final_data["not.fully.paid"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


from sklearn.tree import DecisionTreeClassifier
dtree =  DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300)

rfc.fit(X_train, y_train)

predictions2 = rfc.predict(X_test)

print(classification_report(y_test, predictions2))
print(confusion_matrix(y_test, predictions2))

