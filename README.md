## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed by: Shehan Shajahan
Register No: 212223240154
```
```
import pandas as pd
df = pd.read_csv("Encoding Data.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/7531f5bf-71cc-49c9-a8f6-5ff8ce3fe7c9)
```
df.tail()
```
![image](https://github.com/user-attachments/assets/9903fdd3-25d0-4978-9ffe-be61b684b960)
```
df.describe()
```
![image](https://github.com/user-attachments/assets/78205313-62cb-498b-87f2-e2e516102845)
```
df.info()
```
![image](https://github.com/user-attachments/assets/afc508b6-486c-417e-969e-74cdbc032002)
```
df.shape
```
![image](https://github.com/user-attachments/assets/88683656-a9cb-4917-b9e8-bc908f880dae)
```
df
```
![image](https://github.com/user-attachments/assets/80e704a0-64f5-43bb-bc51-dbad5050cf45)
```
#ordinal encoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/5c103aea-61f5-4b04-bc63-1a4de25f0c41)
```
df['bo2']=oe.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/561d5492-56c5-4e2c-981e-488c8b74c411)
```
#label Encoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/a720eab8-1b86-4904-8bca-4d380975161a)
```
#One hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/ac98867c-8883-4a48-bdb4-4468557a58c7)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/01b853b5-b29a-4c08-8ef4-56326a9fbaa3)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/00b518e6-2cd2-4106-bde2-9576ddd35ec9)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/2ee8c5cb-7e10-498e-846d-89e2f884150a)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/1f780e63-299f-4c93-aad6-f7909314fb01)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/3940dad4-6f8c-41bc-bcbb-990c73a984c3)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/110790c5-5588-4fce-8e3c-821654097612)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c3284a53-1cde-4297-9864-8557e5421c29)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/cf03f1a7-713c-407c-ad85-95d95c44988d)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/0c95cce9-b686-4e4b-8797-2ec0b48bd8a3)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/211e76fe-ad93-4b94-9396-c99a0c542262)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/4ddc3b3c-e0a2-4ee7-bdb5-75e967fac832)
```
df["Moderate Negative Skew_yeojohnson"],parameters =stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/87786dc5-4777-41de-b746-85273892cb78)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c3916c62-8d7d-4196-91b5-c65790dbc032)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])


sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4b0296ba-87a6-45b1-b5b8-0600bad35d03)
```
df
```
![image](https://github.com/user-attachments/assets/9f256fb0-5f73-46e3-af71-6b7a472e6dc5)
```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/974d1346-17af-4625-bf57-25aef8881d0c)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f42f7974-87d3-4a3b-a76a-bc678678e4e0)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/39c1df12-cb19-4685-bb53-1b53e8db2ed2)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/83162936-a989-4ecb-b957-ceffc10d10db)

# RESULT:
Thus in the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
