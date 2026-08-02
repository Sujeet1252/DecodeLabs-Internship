#!/usr/bin/env python
# coding: utf-8

# # Advance EDA & Feature Engineering

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Dataset for Data Analytics - Sheet1.csv")
df.head()


# In[3]:


print("Rows :", df.shape[0])
print("Columns :", df.shape[1])

df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


print("Duplicate Rows :", df.duplicated().sum())


# In[9]:


for col in df.columns:
    print(col, ":", df[col].nunique())


# In[10]:


df.dtypes


# In[11]:


df_clean = df.copy()

df_clean.head()


# In[13]:


missing = df_clean.isnull().sum()
missing


# In[16]:


plt.figure(figsize=(8,4))

sns.heatmap(df_clean.isnull(),
            cbar=False,
            cmap="viridis")

plt.title("Missing Values Heatmap")

plt.show()


# In[17]:


df_clean["Date"] = pd.to_datetime(df_clean["Date"])


# In[18]:


df_clean.info()


# In[19]:


df_clean["Year"] = df_clean["Date"].dt.year

df_clean["Month"] = df_clean["Date"].dt.month_name()

df_clean["Day"] = df_clean["Date"].dt.day

df_clean["Weekday"] = df_clean["Date"].dt.day_name()


# In[20]:


df_clean.head()


# In[21]:


print("Shape :", df_clean.shape)

print("\nMissing Values\n")

print(df_clean.isnull().sum())


# In[22]:


plt.figure(figsize=(8,5))

sns.histplot(df_clean["TotalPrice"], bins=30, kde=True)

plt.title("Distribution of Total Order Price")
plt.xlabel("Total Price")
plt.ylabel("Number of Orders")

plt.show()


# In[23]:


plt.figure(figsize=(6,4))

sns.countplot(
    x="OrderStatus",
    data=df_clean,
    order=df_clean["OrderStatus"].value_counts().index
)

plt.title("Order Status Distribution")
plt.xticks(rotation=20)

plt.show()


# In[24]:


top_products = df_clean["Product"].value_counts().head(10)

plt.figure(figsize=(10,5))

sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title("Top 10 Selling Products")
plt.xlabel("Orders")
plt.ylabel("Product")

plt.show()


# In[25]:


plt.figure(figsize=(6,4))

sns.countplot(
    x="PaymentMethod",
    data=df_clean,
    order=df_clean["PaymentMethod"].value_counts().index
)

plt.title("Payment Methods Used")
plt.xticks(rotation=20)

plt.show()


# In[26]:


monthly_sales = (
    df_clean.groupby("Month")["TotalPrice"]
    .sum()
    .reindex([
        "January","February","March","April",
        "May","June","July","August",
        "September","October","November","December"
    ])
)

plt.figure(figsize=(10,5))

monthly_sales.plot(marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid(True)

plt.show()


# In[27]:


plt.figure(figsize=(7,4))

sns.boxplot(x=df_clean["ItemsInCart"])

plt.title("Items in Cart")

plt.show()


# In[28]:


plt.figure(figsize=(7,5))

sns.scatterplot(
    x="Quantity",
    y="TotalPrice",
    data=df_clean
)

plt.title("Quantity vs Total Price")

plt.show()


# In[29]:


plt.figure(figsize=(8,4))

sns.countplot(
    x="ReferralSource",
    data=df_clean,
    order=df_clean["ReferralSource"].value_counts().index
)

plt.title("Referral Sources")
plt.xticks(rotation=45)

plt.show()


# In[30]:


plt.figure(figsize=(10,7))

numeric_df = df_clean.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# In[31]:


payment_sales = (
    df_clean.groupby("PaymentMethod")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(7,4))

sns.barplot(
    x=payment_sales.index,
    y=payment_sales.values
)

plt.title("Total Sales by Payment Method")

plt.show()


# In[32]:


df_clean["AveragePricePerItem"] = (
    df_clean["TotalPrice"] / df_clean["Quantity"]
).round(2)

df_clean.head()


# In[33]:


df_clean["CouponUsed"] = np.where(
    df_clean["CouponCode"].isnull(),
    "No",
    "Yes"
)

df_clean.head()


# In[34]:


df_clean["WeekendOrder"] = np.where(
    df_clean["Weekday"].isin(["Saturday","Sunday"]),
    1,
    0
)

df_clean.head()


# In[35]:


df_clean.to_csv(
    "cleaned_dataset.csv",
    index=False
)

print("Dataset saved successfully!")


# In[36]:


df_clean.info()


# In[37]:


print("Original Shape :", df.shape)

print("Cleaned Shape :", df_clean.shape)


# In[ ]:




