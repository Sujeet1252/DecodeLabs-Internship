#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset for Data Analytics - Sheet1.csv")

print("Dataset Shape:", df.shape)
df.head()


# In[2]:


df.info()


# In[3]:


missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

print("Missing Values:")
print(missing_values)


# In[4]:


duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)


# In[5]:


df.describe(include="all").T


# In[6]:


categorical_columns = [
    "Product",
    "PaymentMethod",
    "OrderStatus",
    "CouponCode",
    "ReferralSource"
]

for column in categorical_columns:
    print(f"\n--- {column} ---")
    print(df[column].value_counts(dropna=False))


# In[7]:


numerical_columns = [
    "Quantity",
    "UnitPrice",
    "ItemsInCart",
    "TotalPrice"
]

df[numerical_columns].describe()


# In[8]:


plt.figure(figsize=(6, 4))

sns.countplot(x="OrderStatus", data=df)

plt.title("Order Status Distribution")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=20)
plt.show()


# In[9]:


df["OrderStatus"].value_counts()


# In[10]:


plt.figure(figsize=(7, 4))

sns.countplot(x="PaymentMethod", data=df)

plt.title("Payment Method Distribution")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")
plt.xticks(rotation=20)
plt.show()


# In[11]:


df["PaymentMethod"].value_counts()


# In[12]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["TotalPrice"], kde=True, ax=axes[0])
axes[0].set_title("Total Price Distribution")

sns.boxplot(x=df["TotalPrice"], ax=axes[1])
axes[1].set_title("Total Price Boxplot")

plt.tight_layout()
plt.show()


# In[13]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["Quantity"], discrete=True, ax=axes[0])
axes[0].set_title("Quantity Distribution")

sns.histplot(df["ItemsInCart"], discrete=True, ax=axes[1])
axes[1].set_title("Items in Cart Distribution")

plt.tight_layout()
plt.show()


# In[14]:


# High-value transaction indicator
q1 = df["TotalPrice"].quantile(0.25)
q3 = df["TotalPrice"].quantile(0.75)

iqr = q3 - q1
high_value_threshold = q3 + 1.5 * iqr

df["HighValue"] = (
    df["TotalPrice"] > high_value_threshold
).astype(int)


# Large quantity indicator
df["LargeQuantity"] = (
    df["Quantity"] >= df["Quantity"].quantile(0.90)
).astype(int)


# Large cart indicator
df["LargeCart"] = (
    df["ItemsInCart"] >= df["ItemsInCart"].quantile(0.90)
).astype(int)


# Missing coupon indicator
df["NoCoupon"] = (
    df["CouponCode"].isna()
).astype(int)


# In[15]:


df["RiskScore"] = (
    df["HighValue"]
    + df["LargeQuantity"]
    + df["LargeCart"]
    + df["NoCoupon"]
)


# In[16]:


df["RiskScore"].value_counts().sort_index()


# In[17]:


df["FraudRisk"] = (
    df["RiskScore"] >= 3
).astype(int)


# In[18]:


print(df["FraudRisk"].value_counts())

print("\nClass Percentage:")
print(
    df["FraudRisk"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# In[19]:


plt.figure(figsize=(6, 4))

sns.countplot(
    x="FraudRisk",
    data=df
)

plt.title("Fraud Risk Class Distribution")
plt.xlabel("Fraud Risk (0 = Normal, 1 = High Risk)")
plt.ylabel("Number of Transactions")

plt.show()


# In[20]:


df["RiskScore"].value_counts().sort_index()


# In[21]:


df["FraudRisk"].value_counts()


# In[22]:


# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract useful temporal features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)


# In[23]:


print("Target distribution:")
print(df["FraudRisk"].value_counts())

print("\nTarget percentage:")
print(
    df["FraudRisk"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# In[24]:


columns_to_drop = [
    "FraudRisk",
    "RiskScore",
    "HighValue",
    "LargeQuantity",
    "LargeCart",
    "NoCoupon",
    "OrderID",
    "CustomerID",
    "Date",
    "TrackingNumber",
    "ShippingAddress"
]

X = df.drop(columns=columns_to_drop)

y = df["FraudRisk"]


# In[25]:


print("Feature shape:", X.shape)

print("\nFeatures:")
print(X.columns.tolist())


# In[26]:


categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("Categorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# In[27]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# In[28]:


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[30]:


numeric_transformer_lr = ImbPipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = ImbPipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# In[32]:


from sklearn.pipeline import Pipeline

preprocessor_lr = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer_lr,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# In[33]:


logistic_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor_lr),
        ("smote", SMOTE(random_state=42)),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# In[34]:


logistic_pipeline.fit(X_train, y_train)


# In[35]:


preprocessor_rf = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# In[36]:


random_forest_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor_rf),
        ("smote", SMOTE(random_state=42)),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# In[37]:


random_forest_pipeline.fit(X_train, y_train)


# In[38]:


print("Original training distribution:")
print(y_train.value_counts())


# In[39]:


X_train_prepared = preprocessor_rf.fit_transform(X_train)

X_train_smote, y_train_smote = SMOTE(
    random_state=42
).fit_resample(
    X_train_prepared,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# In[40]:


y_pred_lr = logistic_pipeline.predict(X_test)
y_prob_lr = logistic_pipeline.predict_proba(X_test)[:, 1]

y_pred_rf = random_forest_pipeline.predict(X_test)
y_prob_rf = random_forest_pipeline.predict_proba(X_test)[:, 1]


# In[41]:


from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

precision_lr = precision_score(y_test, y_pred_lr, zero_division=0)
recall_lr = recall_score(y_test, y_pred_lr, zero_division=0)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

print("Logistic Regression")
print("-------------------")
print(f"Precision : {precision_lr:.4f}")
print(f"Recall    : {recall_lr:.4f}")
print(f"ROC-AUC   : {roc_auc_lr:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))


# In[42]:


precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

print("Random Forest")
print("-------------")
print(f"Precision : {precision_rf:.4f}")
print(f"Recall    : {recall_rf:.4f}")
print(f"ROC-AUC   : {roc_auc_rf:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))


# In[43]:


model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Precision": [
        precision_lr,
        precision_rf
    ],
    "Recall": [
        recall_lr,
        recall_rf
    ],
    "ROC-AUC": [
        roc_auc_lr,
        roc_auc_rf
    ]
})

model_results


# In[44]:


model_results.round(4)


# In[45]:


cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[46]:


cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[47]:


from sklearn.metrics import roc_curve

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {roc_auc_lr:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {roc_auc_rf:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.show()


# In[48]:


model_results.to_csv(
    "model_comparison.csv",
    index=False
)

print("Model comparison saved successfully.")


# In[49]:


from sklearn.model_selection import GridSearchCV

logistic_param_grid = {
    "smote__sampling_strategy": [0.5, 1.0],
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": ["liblinear"],
    "classifier__class_weight": [None, "balanced"]
}

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)

logistic_grid.fit(X_train, y_train)


# In[50]:


print("Best Logistic Regression Parameters:")
print(logistic_grid.best_params_)

print("\nBest Cross-Validation ROC-AUC:")
print(round(logistic_grid.best_score_, 4))


# In[51]:


random_forest_param_grid = {
    "smote__sampling_strategy": [0.5, 1.0],
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

random_forest_grid = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=random_forest_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)

random_forest_grid.fit(X_train, y_train)


# In[52]:


print("Best Random Forest Parameters:")
print(random_forest_grid.best_params_)

print("\nBest Cross-Validation ROC-AUC:")
print(round(random_forest_grid.best_score_, 4))


# In[53]:


best_lr = logistic_grid.best_estimator_

y_pred_lr_tuned = best_lr.predict(X_test)
y_prob_lr_tuned = best_lr.predict_proba(X_test)[:, 1]

precision_lr_tuned = precision_score(
    y_test,
    y_pred_lr_tuned,
    zero_division=0
)

recall_lr_tuned = recall_score(
    y_test,
    y_pred_lr_tuned,
    zero_division=0
)

roc_auc_lr_tuned = roc_auc_score(
    y_test,
    y_prob_lr_tuned
)

print("Tuned Logistic Regression")
print("-------------------------")
print(f"Precision : {precision_lr_tuned:.4f}")
print(f"Recall    : {recall_lr_tuned:.4f}")
print(f"ROC-AUC   : {roc_auc_lr_tuned:.4f}")


# In[54]:


best_rf = random_forest_grid.best_estimator_

y_pred_rf_tuned = best_rf.predict(X_test)
y_prob_rf_tuned = best_rf.predict_proba(X_test)[:, 1]

precision_rf_tuned = precision_score(
    y_test,
    y_pred_rf_tuned,
    zero_division=0
)

recall_rf_tuned = recall_score(
    y_test,
    y_pred_rf_tuned,
    zero_division=0
)

roc_auc_rf_tuned = roc_auc_score(
    y_test,
    y_prob_rf_tuned
)

print("Tuned Random Forest")
print("-------------------")
print(f"Precision : {precision_rf_tuned:.4f}")
print(f"Recall    : {recall_rf_tuned:.4f}")
print(f"ROC-AUC   : {roc_auc_rf_tuned:.4f}")


# In[55]:


final_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Tuned Logistic Regression",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "Precision": [
        precision_lr,
        precision_lr_tuned,
        precision_rf,
        precision_rf_tuned
    ],
    "Recall": [
        recall_lr,
        recall_lr_tuned,
        recall_rf,
        recall_rf_tuned
    ],
    "ROC-AUC": [
        roc_auc_lr,
        roc_auc_lr_tuned,
        roc_auc_rf,
        roc_auc_rf_tuned
    ]
})

final_results.round(4)


# In[56]:


final_results.to_csv(
    "final_model_comparison.csv",
    index=False
)


# In[57]:


final_results.sort_values(
    by="ROC-AUC",
    ascending=False
).round(4)


# In[58]:


print("Best Logistic Regression:")
print(logistic_grid.best_params_)

print("\nBest Random Forest:")
print(random_forest_grid.best_params_)


# In[59]:


from sklearn.metrics import roc_curve

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    y_prob_lr_tuned
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_rf_tuned
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {roc_auc_lr_tuned:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {roc_auc_rf_tuned:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Tuned Models")
plt.legend()

plt.show()


# In[60]:


cm = confusion_matrix(
    y_test,
    y_pred_rf_tuned
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Tuned Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[61]:


cm = confusion_matrix(
    y_test,
    y_pred_lr_tuned
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Tuned Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[62]:


rf_preprocessor = best_rf.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()

rf_model = best_rf.named_steps["classifier"]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.head(15)


# In[63]:


top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Features — Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.show()


# In[64]:


feature_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)


# In[65]:


plt.savefig(
    "roc_curve_tuned_models.png",
    dpi=300,
    bbox_inches="tight"
)


# In[ ]:




