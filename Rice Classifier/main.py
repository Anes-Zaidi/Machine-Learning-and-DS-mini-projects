from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


riceData = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

model = LogisticRegression()

scaler = StandardScaler()

# We have two types of rice Cammeo and Osmancik
# We are going to classify these types into two classes 1 & 0.
riceData["Class"] = riceData["Class"].map({"Cammeo" : 1 , "Osmancik" : 0})

# We are dropping "Minor_Axis_Length" and "Extent" features because they have a low correalation coefiecent with
#   our label ("Class")
# You can check the correaltion matrix yourself using : riceData.corr()
# Or use seaborn to plot the correaltion,
#         corr = riceData.corr()
#         sns.heatmap(corr, annot=true, cmap="coolwarm")
#         plt.title("Correlation Matrix")
#         plt.show()

riceData = riceData.drop(["Minor_Axis_Length" , "Extent"] , axis=1)

label = riceData.loc[: , "Class"]
features = riceData.drop("Class" , axis=1)

# Normalize our data
ScaledFeatures = scaler.fit_transform(features)

# Since StandardScaler returns numpy array we are going to convert the data to a DataFrame again 
ScaledFeatures = pd.DataFrame(ScaledFeatures , columns=features.columns)


#Split the data into test set & training set
X_train , X_test , Y_train , Y_test = train_test_split(ScaledFeatures, label , test_size=0.25 , stratify=label)

# Train the model
model.fit(X_train,Y_train)

# Get the model predictions
y_pred = model.predict(X_test)

# Mesuring our model performance :


print(classification_report(Y_test,y_pred))

# Plot confusion matrix
cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm , annot=True , fmt="d" , cmap="Blues", xticklabels=["Osmancik", "Cammeo"], yticklabels=["Osmancik", "Cammeo"])

plt.xlabel("Pridected")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()




