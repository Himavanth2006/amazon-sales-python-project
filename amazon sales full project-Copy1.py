#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv(r"C:\Users\himav\Downloads\amazon data.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.dtypes


# In[7]:


df["product_id"].unique()


# In[8]:


df["product_name"].unique()


# In[9]:


df["category"].unique()


# In[10]:


df["discounted_price"].unique()


# In[11]:


df["actual_price"].unique()


# In[12]:


df['discount_percentage'].unique()


# In[13]:


df["rating"].unique()


# In[14]:


df["rating_count"].unique()


# In[15]:


# Clean discounted_price
df['discounted_price'] = df['discounted_price'].astype(str)
df['discounted_price'] = df['discounted_price'].str.replace('₹','', regex=False)
df['discounted_price'] = df['discounted_price'].str.replace(',','', regex=False)
df['discounted_price'] = df['discounted_price'].astype(float)


# In[16]:


# Clean discounted_price
df['discounted_price'] = df['discounted_price'].astype(str)
df['discounted_price'] = df['discounted_price'].str.replace('₹','', regex=False)
df['discounted_price'] = df['discounted_price'].str.replace(',','', regex=False)
df['discounted_price'] = df['discounted_price'].astype(float)


# In[17]:


# Clean discount_percentage
df['discount_percentage'] = df['discount_percentage'].astype(str)
df['discount_percentage'] = df['discount_percentage'].str.replace('%','', regex=False)
df['discount_percentage'] = df['discount_percentage'].astype(float)


# In[18]:


# Clean rating
df['rating'] = df['rating'].astype(str)
df['rating'] = df['rating'].str.extract('(\d+\.?\d*)')
df['rating'] = df['rating'].astype(float)


# In[19]:


# Clean rating_count
df['rating_count'] = df['rating_count'].astype(str)
df['rating_count'] = df['rating_count'].str.replace(',','', regex=False)
df['rating_count'] = df['rating_count'].astype(float)


# In[20]:


#coverted the data into float
df.info()


# In[21]:


#coverted the data into float
df.dtypes


# In[22]:


# checking null values
df.isnull().sum()


# In[23]:


#searching for the null values
df[df.isnull().any(axis=1)]


# In[24]:


# Fill null values in numeric columns with median( TO FILL NUMERIC VALUES USE MEDIAN AND FOR OBJECT USE MODE)

df['rating'] = df['rating'].fillna(df['rating'].median())

df['rating_count'] = df['rating_count'].fillna(df['rating_count'].median())


# In[25]:


# agian checking if the null values exsist
df.isnull().sum()


# In[26]:


df.info()


# In[27]:


# saving the cleaned data 
df.to_csv("amazon_cleaned_final.csv", index=False)
print("File saved successfully")


# In[28]:


# Summary statistics of numeric columns
df.describe()


# In[29]:


# Discounted Price Analysis:
# The average price is ₹3125 and median is ₹799, indicating right-skewed distribution.
# The high standard deviation (6944) shows large variation in product prices.

# Discount Percentage Analysis:
# The average discount is 47.69% and median is 50%, indicating most products offer high discounts.
# The standard deviation (21.63) shows moderate variation in discount values.

# Rating Analysis:
# The average rating is 4.09 and median is 4.1, indicating generally high ratings.
# The low standard deviation (0.29) shows ratings are consistent.

# Rating Count Analysis:
# The average rating count is 18,277 and median is 5,179, indicating right-skewed distribution.
# The high standard deviation (42,727) shows large variation in product popularity.


# In[32]:


# Summary statistics of categorical columns
categorical = ["product_name", "category"]
df[categorical].describe()


# In[33]:


# Categorical Columns Analysis:
# The dataset contains 1337 unique product names and 211 unique categories.
# This indicates most products are unique and belong to a wide range of categories.
# The most frequent category appears 233 times, showing some categories have more products than others.


# In[35]:


df.duplicated().sum()


# In[36]:


# Duplicate Rows Analysis:
# No duplicate rows were found in the dataset.
# This indicates the dataset is clean and does not contain repeated records.


# In[37]:


# Skewness of numeric columns
df.skew(numeric_only=True)


# In[39]:


# Skewness Analysis: observations

# Discounted Price Analysis:
# The skewness value is 4.45, indicating highly positive skewness.
# This means most products have lower prices, with few very expensive products.

# Discount Percentage Analysis:
# The skewness value is -0.29, indicating slight negative skewness.
# This means discount values are fairly evenly distributed.

# Rating Analysis:
# The skewness value is -1.24, indicating negative skewness.
# This means most products have higher ratings, with fewer low-rated products.

# Rating Count Analysis:
# The skewness value is 5.67, indicating highly positive skewness.
# This means most products have fewer ratings, while few products are extremely popular.


# In[40]:


# Histogram of discounted price to visualize the distribution of product prices
plt.hist(df["discounted_price"])
plt.title("Distribution of Discounted Price")
plt.xlabel("Discounted Price")
plt.ylabel("Number of Products")
plt.show()


# In[41]:


# Observation:
# Most products are concentrated in the lower price ranges,
# while few products have very high prices, indicating positive skewness in the data.


# In[42]:


# Histogram of discount percentage to visualize discount distribution
plt.hist(df["discount_percentage"])
plt.title("Distribution of Discount Percentage")
plt.xlabel("Discount Percentage")
plt.ylabel("Number of Products")
plt.show()


# In[44]:


# Observation:
# Most products have discount percentages between 40% and 70%,
# indicating that moderate to high discounts are common.
# Very few products have extremely low or extremely high discounts.


# In[45]:


# Histogram of rating to visualize the distribution of product ratings
plt.hist(df["rating"])
plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Products")
plt.show()


# In[46]:


# Observation:
# Most products have ratings between 4.0 and 4.5, indicating high customer satisfaction.
# Very few products have ratings below 3.5, showing most products are well-rated.


# In[47]:


# Histogram of rating count to visualize the distribution of product popularity
plt.hist(df["rating_count"])
plt.title("Distribution of Rating Count")
plt.xlabel("Rating Count")
plt.ylabel("Number of Products")
plt.show()


# In[48]:


# Observation:
# Most products have low rating counts, as shown by the tall bars on the left side.
# Very few products have extremely high rating counts, indicating that only a small number of products are highly popular.
# This confirms a strong positive skew in product popularity.


# In[50]:


# Correlation matrix to check relationships between numeric variables
df.corr(numeric_only=True)


# In[52]:


# Correlation Analysis:
# The correlation matrix shows weak relationships between most variables.
# Discounted price and discount percentage have slight negative correlation.
# Rating and rating count have very weak positive correlation, indicating popular products tend to have slightly higher ratings.
# Overall, price, discount, and rating are mostly independent variables.


# In[ ]:





# In[ ]:


# Final Insights from Amazon Product Data Analysis:

# 1. Pricing Insight:
# Most products are priced in the lower range, with a median price of ₹799.
# However, a few products have very high prices, which increases the overall average price.

# 2. Discount Insight:
# Most products offer discounts between 40% and 70%, indicating that Amazon frequently uses discounts
# as a strategy to attract customers.

# 3. Rating Insight:
# Most products have ratings between 4.0 and 4.5, showing that customer satisfaction is generally high.

# 4. Popularity Insight:
# Most products have relatively low rating counts, while a small number of products have extremely high rating counts.
# This indicates that only a few products are highly popular among customers.

# 5. Category Insight:
# The dataset contains a large number of unique categories, showing that Amazon offers a wide variety of products.
# Some categories contain significantly more products than others.

# 6. Overall Business Insight:
# The analysis shows that most Amazon products are affordable, highly rated, and moderately discounted.
# Product popularity is uneven, with a small number of products receiving most of the customer attention.


# In[54]:


# Save the final cleaned and analyzed dataset
df.to_csv("amazon_final_dataset.csv full project", index=False)


# In[ ]:




