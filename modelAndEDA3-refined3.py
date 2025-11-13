#%% md
# https://www.kaggle.com/datasets/warcoder/earthquake-dataset?select=earthquake_1995-2023.csv
# 
# I need to do an appendix for my references?
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
#%% md
# ### Dataset Background and Source
# 
# The Earthquake dataset was sourced from the United States Geological Survey (USGS) earthquake everywhere API and uploaded to Kaggle by Chirag Chauhan. It comprises data on 782 significant global earthquakes from 2001–2022 which supports research in earthquake analysis, tsunami risk prediction and seismic hazard assessment (Chauhan, 2024)
# 
# 
# ### My goal
# 
# My goal is to predict the occurrence of tsunamis following earthquakes based on the characteristics of the seismic events.
# 
# #### Stakeholders and Audience
# 
# If I'm able to model and predict tsunami occurrence accurately, it could have significant implications for disaster preparedness and mitigation.
# 
# Stakeholders who could significantly benefit from this analysis include:
# - Goverment departments responsible for disaster management and emergency response
# - Coastal city planners and infrastructure developers
# - Environmental and geological research teams and institutions
# 
# 
# #### Research Questions
# 
# Will decision tree models outperform logistic regression in predicting tsunami occurrence following earthquakes based on seismic event characteristics?
# 
# Which feature will be the most important predictor of tsunami occurrence following earthquakes?
# 
# 
# 
# ### Feature Overview
# 
# | Column       | Description                                                                                                                                            |
# |--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
# | title        | Name given to earthquake                                                                                                                     |
# | magnitude    | Magnitude of earthquake                                                                                                                        |
# | date_time    | Date and time                                                                                                                                          |
# | cdi          | Maximum reported intensity for event range                                                                                                   |
# | mmi          | Maximum estimated instrumental intensity for  event                                                                                             |
# | alert        | Alert level - “green”, “yellow”, “orange”, and “red”                                                                                                |
# | tsunami      | "1" for events in oceanic regions and "0" otherwise                                                                                                     |
# | sig          | Number describing how significant the event is. Larger numbers indicate a more significant event. This value is determined by magnitude, maximum MMI, felt reports, and estimated impact |
# | net          | ID of a data contributor. Identifies the network considered to be the preferred source of information for this event.                               |
# | nst          | Total number of seismic stations used to determine earthquake location.                                                                            |
# | dmin         | Horizontal distance from the epicenter to the nearest station                                                                                          |
# | gap          | Largest azimuthal gap between azimuthally adjacent stations (in degrees). Smaller numbers indicate more reliable positions. Gaps >180° have large uncertainties |
# | magType      | Method or algorithm used to calculate the preferred magnitude for the event                                                                       |
# | depth        | Depth where the earthquake begins to rupture                                                                                                        |
# | latitude     | Latitude coordinate of the earthquake                                                                                                                  |
# | longitude    | Longitude coordinate of the earthquake                                                                                                                 |
# | location     | Location within the country                                                                                                                            |
# | continent    | Continent of the earthquake-hit country                                                                                                                |
# | country      | Affected country                                                                                                                                       |
# 
# (Chauhan, 2024)
# 
# ### Expected relationships within the data:
# 
# Before analyzing the data, we can hypothesise some expected relationships:
# 
# - Higher magnitude likely means higher significance (sig)
# - Higher magnitude likely means higher intensity (cdi, mmi)
# - Higher magnitude likely means higher tsunami risk
# - Geographical location (latitude, longitude) likely influences tsunami risk based on proximity to coastlines
# - Alert levels likely correlate with magnitude, intensity, and significancee
#%%
df = pd.read_csv('data/earthquake_data.csv')
#%%
df.hist(bins=15, figsize=(15,10), layout=(4,4))
plt.tight_layout()
plt.show()
#%% md
# Figure 1: Histograms of Numerical Features
# 
# This histogram visualization reveals several important insights about the dataset.
# - tsunami is a binary variable with most values being 0 (no tsunami).
# - left skewed distributions appear common for `magnitude, sig, cdi, mmi, nst, dmin, gap, and depth.`
#     - This suggests that most earthquakes in this dataset are of lower magnitude and general impact with a smaller collection of high magnitude and high impact events, these are likely to be the larger scale earthquakes.
#%%
df.sample(16)
#%% md
# #### Features / Feature columns
# 
# `magnitude`, `cdi`, `mmi`, `sig`, `nst`, `dmin`, `gap`, `depth`, `latitude`, `longitude`,
# `title`, `date_time`, `alert`, `net`, `magType`, `location`, `continent`, `country`
# 
# Further analysis is needed to confirm these feature types and their relevance to tsunami prediction.
# 
# ---
# 
# #### Target Variable / Target Column
# 
# **`tsunami`** – Tsunami occurrence indicator (`1 = tsunami`, `0 = no tsunami`)
# 
# Our target binary numerical variable represents whether a tsunami occurred following an earthquake event. The importance of this target If we model and predict this variable accurately, it could have significant implications for disaster preparedness and mitigation. For example, if we can predict tsunami occurrence based on earthquake characteristics, we can issue timely warnings to coastal communities, potentially saving lives and reducing property damage.
#%% md
# #### Statistical Summary
# 
#%%
df.describe()
#%%
print(f'Dataset Shape: {df.shape}')
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print('Column Names:', df.columns.tolist())
#%% md
# ### Data Types
# 
#%%
unique_values = df.nunique()
print(f'Unique values per column:\n{unique_values}')
country_unique_variables_before_imputation = df['country'].nunique()
#%%
df.nunique(axis=0).plot(kind='bar')
plt.title('Unique Value Counts per Column')
plt.xlabel('Columns')
plt.ylabel('Unique Value Counts')
plt.show()
#%% md
# Figure 2: This chart shows the number of unique values for each dataset column. This is useful for identifying categorical features (which typically have a limited number of unique values) versus numerical features (which often have a larger number of unique values).
#%%
print(df.dtypes)
#%% md
# The 'object' indicates string/categorical data types in pandas DataFrames, while 'float64' and 'int64' indicate numerical data types.
#%% md
# When deciding if a variable is categorical or numerical we considered that:
# - Numerical variables typically have a large number of unique values (e.g., more than 15-20 unique values).
# - Categorical variables usually have a limited number of unique values (e.g., less than 15-20 unique values) and often represent distinct categories or groups.
# - Categorical variables can also be identified by their data type (e.g., 'object' in pandas).
# - Textual data can be identified by sampling the data and looking at the values.
# 
# Numerical datatypes, `sig`,`nst`,`dmin`,`gap`,`depth`,`latitude` and `longitude` have a large number of unique values. However, features like `magnitude`, `cdi`, and `mmi` are numerical but have a limited number of unique values (as per figure 2) so could be treated as numerical or categorical. According to the USGS, `mmi` and `cdi` are both intensity scales that measure the effects of an earthquake, while `magnitude` measures the energy released. `magnitude` will therefore be treated numerical as it represents a continuous scale of energy release but `cdi` and `mmi` will be treated as ordered categorical features as they represent discrete intensity levels.
# (United States Geological Survey, no date; Wald et al., 2011; United States Geological Survey, 1989)
# 
# 
# `country`, `contient`, `magType`,`alert`and `net` are categorical text features because they are of object data type and represent distinct categories shown by their lower relative unique value counts. `title` and `location` are unstructured text features because they are of object data type but have a higher relative unique value counts (shown in bar chart) of more free-form text data. `date_time` is a datetime feature.  Our target variable `tsunami` is a binary numerical variable having only has 2 unique values (0 and 1).
# 
# 
# 
# 
# | Types of data               | Features                                               |
# |-----------------------------|--------------------------------------------------------|
# | Numerical Features          | `magnitude`, `sig`, `nst`, `dmin`, `gap`, `depth`, `latitude`, `longitude` |
# | Categorical Features        | `cdi`, `mmi`                                          |
# | Datetime Feature            | `date_time`                                           |
# | Categorical Text Feature   | `alert`, `net`, `magType`, `continent`, `country`          |
# | Unstructured Text Feature   | `title`, `location`                                   |
# | Target Variable (Binary Numerical) | `tsunami`                                             |
# 
# Figure 3: Data Types of Features and Target Variable
#%% md
# ### Mapping my task to ML problem type
# 
# The goal is to predict the occurrence of tsunamis following earthquakes based on the characteristics of the seismic events. Given the target variable `tsunami` is binary (1 = tsunami, 0 = no tsunami), indicating whether a tsunami occurred after an earthquake event, this is a **binary classification** problem. Using the features of the earthquakes we will classify whether a tsunami will occur (1) or not (0) following an earthquake event.
# 
# #### Key features that will likely be important for predicting tsunami occurrence and there meaning
# 
# Observing (Chauhan, 2025, About Dataset) and applying general domain knowledge about earthquakes and tsunamis, the following features stand out as particularly relevant for predicting tsunamis :
# 
# - `magnitude`: Higher magnitude earthquakes are more likely to generate tsunamis.
# - `latitude` and `longitude`: The geographical location of the earthquake can influence tsunami risk, especially near coastal areas.
# - `cdi` and `mmi`: Higher intensity levels may correlate with tsunami generation.
# - `sig`: More significant earthquakes may have a higher likelihood of causing tsunamis.
# 
# 
#%% md
# ### Missing Values
# 
#%%
print(df.isnull().sum())
#%%
place_holders = [-9999, 9999, np.inf, -np.inf, 'nan', 'NaN', 'NA', 'null', 'NULL', '']
print(df.isin(place_holders).sum().sum() + df.isnull().sum().sum())
#%%
#percentage of missing values per column
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_percentage = missing_percentage[missing_percentage > 0]
print(f'Percentage of missing values per column:\n{missing_percentage}')
#%%
missing_values_datatypes = df[missing_percentage.index].dtypes
print(f'Data types of columns with missing values:\n{missing_values_datatypes}')
#%%
df[missing_percentage.index].sample(16)
#%% md
# Object in this context refers to string/categorical data types in pandas DataFrames.
# 
# Referencing figure 3, the columns missing values are either unstructured text features (`location`, `country`) or categorical text features (`alert`,`continent`). There are no missing values in numerical features.
# 
# #### Understanding missing data patterns
# 
# | **Feature**              | **Type of Missingness**           | **Reason / Explanation**                                                                                                                                                                                                                 |
# |---------------------------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | **Alert Level**           | Missing Not At Random (MNAR)     | Missingness is related to the unobserved value itself — less significant earthquakes are less likely to have an alert level assigned.                                                              |
# | **Location**              | Missing At Random (MAR)          | Missingness depends on other observed variables (latitude and longitude). Location may not be recorded because coordinates are already available.                                                  |
# | **Country**               | Missing At Random (MAR)          | Missingness is related to latitude and longitude data — the country may be inferred from coordinates, making direct recording unnecessary.                                                         |
# | **Continent**             | Missing At Random (MAR)          | Missingness depends on other observed information (such as latitude, longitude, or country) and is not directly related to the continent value itself.                                              |
# 
# 
#%% md
# #### Alert missing values imputation strategy
#%%
#get unique values for alert column
df['alert'].value_counts().plot(kind='bar')
#%% md
# This charft tells us a few important things about the `alert` column:
# - The most common alert level is "green", indicating low severity earthquakes.
# - It has a small number of fixed categories (green, yellow, orange, red) which should make imputation easier.
#%% md
# If we encode the alert level from categorical to numerical, we can study it in a correlation matrix to see how other features relate to it and possibly use those relationships to impute missing alert values. This follows because alert levels are assigned based on earthquake characteristics like magnitude, intensity, and significance which are all numerical features.
# 
# As shown in figure 3, `alert` is a categorical text feature with 4 unique levels: green, yellow, orange, and red with different frequencies. The dataset has a left skewed distribution of earthquake magnitudes and impacts (figure 1), hence the lower severity alert levels (green and yellow) are more common than higher severity levels (orange and red). This strongly suggests that alert levels are related to earthquake characteristics which we can leverage for imputation.
# 
# `alert` is an ordinal categorical variable where the levels have a natural order of severity: green < yellow < orange < red (United States Geological Survey, no date, section “PAGER Scientific Background”).
# 
# 
#%%
#ordinal encoding of alert levels
alert_order = {'green': 1, 'yellow': 2, 'orange': 3, 'red': 4}
df_temp = df.copy()
df_temp['alert_encoded'] = df_temp['alert'].map(alert_order)
numerical_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df_temp[numerical_cols].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()
#%% md
# Studying the correlation matrix, we can see there is a strong correlation between `alert` and `sig` (0.79) so we will use this correlation to impute missing alert levels based on the `sig` value of each earthquake event.
# 
# This will be a rule based / conditional imputation strategy where we calculate the mean `sig` value for each alert level and use those thresholds to impute missing alert levels based on the `sig` value of each earthquake event.
#%%
print(df.groupby('alert').agg({
    'sig': ['mean']
}).sig)
#%%
alert_thresholds = df.groupby('alert').agg({
    'sig': ['mean']
}).sig

green_threshold = alert_thresholds.loc['green', 'mean'] #index by alert level, get column 'mean'
yellow_threshold = alert_thresholds.loc['yellow', 'mean']
orange_threshold = alert_thresholds.loc['orange', 'mean']
red_threshold = alert_thresholds.loc['red', 'mean']

#Order of alert levels: red, orange, yellow, green - source: https://earthquake.usgs.gov/data/pager/background.php

def impute_alert(row):
    if pd.isnull(row['alert']):
        if row['magnitude'] >= red_threshold:
            return 'red'
        elif row['magnitude'] >= orange_threshold:
            return 'orange'
        elif row['magnitude'] >= yellow_threshold:
            return 'yellow'
        else:
            return 'green'
    else:
        return row['alert']
#apply the imputation function
df['alert'] = df.apply(impute_alert, axis=1)
#verify no more missing values in alert column
print(f'Missing values in alert column after imputation: {df["alert"].isnull().sum()}')
#%% md
# #### Location missing values imputation strategy
#%%
msno.matrix(df)
#%% md
# `Location` is closely related to the `country` feature as it describes the area within the country. `location` is an unstructured text feature and `country` categorical text feature and, given they are the same type of data, we can use one to impute missing values in the other.
# 
# The true location values will have a more precise description of the area within the country but, using this strategy, is better than having no location information at all. This is possible because the missing matrix shows that in many cases where `location` is missing, `country` is present.
# 
# 
# This is another example of rule based / conditional imputation but this time based on the categorical relationship between the `location` and `country` features.
# 
# 
#%%
print(f'Missing values in location column before imputation: {df["location"].isnull().sum()}')
def impute_location(row):
    if pd.isnull(row['location']):
        if not pd.isnull(row['country']):
            return f'{row["country"]}'
        else:
            return np.nan # cannot impute location
    else:
        return row['location']
#apply the imputation function
df['location'] = df.apply(impute_location, axis=1)
#verify no more missing values in location column
print(f'Missing values in location column after imputation: {df["location"].isnull().sum()}')
#%%
df = df.dropna(subset=['location'])
print('Data shape after dropping rows with missing location:', df.shape)
#%% md
# The remaining missing values in the location column are those where both location and country were missing so we could not impute anything meaningful and have dropped these rows from the dataset. This will remove 3 rows from the dataset, leaving us with 779 samples.
#%%
print(df.location.sample(15))
#%% md
# #### Country missing values imputation strategy
#%%
coma_count = df['location'].str.contains(',').sum()
print(f'Percentage of locations with a comma: {(coma_count / len(df)) * 100:.2f}%')
#%% md
# The random samples above show many of the location entries contain a comma, indicating both a specific area and the country name (e.g., "Los Angeles, USA"). This infomation can help us populate missing country values from the location column by taking the substring after the comma as the country and, given the quantity of missing country values (298), this will introduce some multi-collinearity.
# 
# I believe this is acceptable as having some country information is better than none at all for our modeling and, similar to using the `country` feature to impute missing `location` values, we can use the `location` feature to impute missing `country` values.
# 
# This is another example of rule based / conditional imputation based on the relationship between the `location` and `country` features.
#%%
print(f'Missing values in country column before imputation: {df["country"].isnull().sum()}')
def impute_country(row):
    if pd.isnull(row['country']):
        if ',' in row['location']:
            return row['location'].split(',')[-1].strip() # extract country from location using the last part after comma
        else:
            return np.nan # cannot impute country
    else:
        return row['country']
#apply the imputation function

#verify no more missing values in country column
df['country'] = df.apply(impute_country, axis=1)
print(f'Missing values in country column after imputation: {df["country"].isnull().sum()}')
#%%
missing_country_locations = df[df['country'].isnull()]['location']
print(f'Locations with missing country values:\n{missing_country_locations}')
#%% md
# Many of the 42 samples without country data are small islands, territories, ocean regions, or disputed areas.
# 
# Having analysed this data, i'm confident these values represent similar value to countries and will therefore impute these missing country values with the location value itself for these instances.
# 
#%%
df['country'] = df['country'].fillna(df['location'])
#verify no more missing values in country column
print(f'Missing values in country column after final imputation: {df["country"].isnull().sum()}')
#%% md
# #### Continent missing values imputation strategy
#%%
msno.matrix(df[['continent', 'country', 'location']])
#%% md
# Data missing from the 'contient' column is 73%. Given this high level of missingness, it is therefore more practical to drop the 'continent' column entirely from the dataset rather than attempting to impute the missing values. Imputing such a large proportion of missing data could introduce significant bias and uncertainty into the dataset.
# 
#%%
df = df.drop(columns=['continent'])
print(df.columns)
#%% md
# #### Visualizing missing data pattern after imputation
#%%
total_missing_after = df.isnull().sum().sum()
print(f'Total missing values after imputation: {total_missing_after}')
#%%
X, y = df.drop(columns='tsunami'), df['tsunami']
X.shape, y.shape, X.columns
#%% md
# X is our new Dataframe containing all features except the target variable `tsunami`, and y is a Series containing target variable `tsunami` which we will predict.
# 
# X a fully imputed feature set with no missing values, and y `tsunami` didn't have any missing values to begin with.
#%% md
# ### Encoding Categorical Features and Text Features
#%% md
# We must encode categorical features into numerical format before modeling as most machine learning algorithms require numerical input. We will use one-hot encoding for nominal categorical features and ordinal encoding for ordinal categorical features (figure 3)
# 
# During our feature type identification earlier, we determined that `cdi` and `mmi` are ordinal categorical features as they represent intensity levels with a natural order of severity and have a limited number of unique values (United States Geological Survey (USGS), 1989; Wald et al., 2011). These features do not need encoding as they are already in numerical format.
# 
# `alert`, is also an ordinal categorical variable so we will apply ordinal encoding as it is currently in a categorical text format and needs to be converted to numerical format for modeling.
#%%
from sklearn.preprocessing import OrdinalEncoder
ordinal_features = ['alert']
ordinal_encoder = OrdinalEncoder(categories=[['green', 'yellow', 'orange', 'red']])
X[ordinal_features] = ordinal_encoder.fit_transform(X[ordinal_features])
#%% md
# We must also encode our other categorical text features: `net`, `magType`,`Contient`, and `country` (figure 3). These are nominal categorical variables as there is no inherent order to the categories they represent. We will use one-hot encoding to convert these nominal categorical features into numerical format by creating binary indicator variables for each category.
#%%
#one-hot encoding of nominal categorical features
columns_before = X.shape[1]
nominal_features = ['net', 'magType']
X = pd.get_dummies(X, columns=nominal_features, drop_first=True)
columns_after = X.shape[1]
#%%
print(country_unique_variables_before_imputation)
print(df.country.nunique())
plt.bar(['Before Imputation', 'After Imputation'], [country_unique_variables_before_imputation, df.country.nunique()])
plt.title('Unique Country Values Before and After Imputation')
plt.ylabel('Unique Value Count')
plt.show()
#%% md
# Having imputed missing values in the `country` feature earlier its unique value count has grown from 49 to 87 unique values. This is expected as we filled in missing country values using location data, which introduced new country names that were not present before imputation. This means our one-hot encoding of the `country` feature will now create 86 new binary columns (after dropping the first to avoid multicollinearity) instead of 48 as before imputation. To avoid high dimensionality and overfitting, we are going to use target encoding for the `country` feature. This will reduce the number of new features created while still capturing the relationship between country and tsunami occurrences.
# 
# Target encoding involves replacing each category in the `country` feature with the mean of the target variable (`tsunami`) for that category. This way, we convert the categorical `country` feature into a single numerical feature that reflects the average tsunami occurrence rate for each country.
#%%
country_target_mean = y.groupby(X['country']).mean()
X['country_encoded'] = X['country'].map(country_target_mean)
X = X.drop(columns=['country'])
X
#%% md
# Now we must process our unstructured text features: `title` and `location` (figure 3). These features contain free-form text data that cannot be directly used in most ML models so we will need to convert this text data into numerical format using text vectorization techniques.
# 
#%% md
# We’ll use Word2Vec embeddings to convert the title and location text features into numerical vectors. By averaging the word vectors, we create a single representation for each text entry, capturing overall meaning (itdxer, 2017) while limiting dimensionality and avoiding overfitting with only 782 samples. Although this loses some context, it efficiently incorporates text data. We’ll use glove-wiki-gigaword-50, which provides 50-dimensional embeddings that balance semantic detail and computational efficiency.
# 
#%%
import gensim.downloader as api
word2vec_model = api.load('glove-wiki-gigaword-50') #pre-trained Word2Vec model with 50-dimensional vectors

def average_word_vectors(text, model):
    words = str(text).lower().split()
    vectors = []
    for words in words:
        if words in model: #When you iterate through model, you get the words in the vocabulary (it must have a custom __iter__ method)
            vectors.append(model[words])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

#apply the averaging function to title and location columns
title_vectors = X['title'].apply(lambda x: average_word_vectors(x, word2vec_model))
location_vectors = X['location'].apply(lambda x: average_word_vectors(x, word2vec_model))
title_df = pd.DataFrame(title_vectors.tolist(), index=X.index).add_prefix('title_vec_') #take index from original X to align rows
location_df = pd.DataFrame(location_vectors.tolist(), index=X.index).add_prefix('location_vec_')
X = pd.concat([X, title_df, location_df], axis=1)
X = X.drop(columns=['title', 'location'])
X
#%% md
# #### Not all models require only numerical input features.
#  decsion trees and ensemble models can handle categorical features directly. However, many popular machine learning algorithms, such as logistic regression require numerical input features. So in order to do a fair comparision later on between different model types, we will convert all categorical features to numerical format now.
#%%
object_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f'Object type columns in X: {object_columns}')
#%%
X['date_time'] = pd.to_datetime(X['date_time'])
X.date_time.dtype
#%%
object_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f'Object type columns in X: {object_columns}')
#%% md
# ### Further data visualization and issue identification
#%%
non_encoded_render = [c for c in X.columns if '_' not in c] #exclude vectorized text features for boxplots because they have many dimensions
plt.figure(figsize=(15,10))
for i, col in enumerate(non_encoded_render):
    plt.subplot(4, 4, i+1)
    sns.boxplot(y=X[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()
#%% md
# #### Extreme outliers issue
# 
# From these boxplots we can see that several features have significant outliers, including `magnitude`, `sig`, `nst`, `dmin`, `gap`, and `depth`. These outliers could potentially impact model performance, especially for algorithms sensitive to extreme values like logistic regression and neural networks.
#%%
features_of_interest = ['magnitude', 'sig', 'nst', 'dmin', 'gap', 'depth']
plt.figure(figsize=(15,10))
for i, col in enumerate(features_of_interest):
    plt.subplot(4, 4, i + 1)
    sns.violinplot(y=X[col])
    plt.title(f'Violin Plot of {col}')
plt.tight_layout()
plt.show()
#%% md
# Our various violin plots above confirm the presence of extreme outliers in these features, as indicated by the long tails extending from the main distribution. We can see that the majority of data points are concentrated in a smaller range, while a few extreme values stretch far beyond this range, indicating the presence of outliers.
#%%
for col in features_of_interest:
    min_value = X[col].min()
    print(f'Minimum value in {col}: {min_value}')
#%% md
# 
# We will use log1p transformation which is log(1 + x). This is useful because it can handle zero values without resulting in undefined values, and it compresses the scale of the data, reducing the impact of extreme outliers while preserving the relative differences between values.
#%%
X[features_of_interest] = X[features_of_interest].apply(lambda x: np.log1p(x))
plt.figure(figsize=(15,10))
for i, col in enumerate(features_of_interest):
    plt.subplot(4, 4, i + 1)
    sns.violinplot(y=X[col])
    sns.boxplot(y=X[col])
    plt.title(f'Violin Plot of {col} after Log Transformation')
plt.tight_layout()
plt.show()
#%% md
# Figure 3: Violin Plots of Features After Log Transformation
# 
# We can see that after log transformation, the distributions of these features are more compact, and the extreme outliers have been reduced in influence. The main body of the data is now more prominent, with a large portion of the data concentrated in a smaller range, making it easier for models to learn from the data without being overly influenced by extreme values.
# 
# Most of our values are still concentrated towards the lower end of the scale, which is expected given the left-skewed distributions we observed earlier in figure 1. However, the log transformation has helped to mitigate the impact of extreme outliers, making the data more suitable for modeling and addressed this skew to an extent.
#%%
print(X.isnull().sum().sum())
#%%
#Correlation matrix heatmap
corr_matrix = X[non_encoded_render].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix Heatmap')
plt.show()
#%% md
# None of our features seem too highly correlated (above 0.9) so we don't need to drop any features due to multicollinearity.
#%%
#check for duplicate rows
duplicate_rows = X.duplicated().sum()
print(f'Number of duplicate rows in X: {duplicate_rows}')
#%% md
# ### Feature Reduction
# 
#  Given our small dataset size of 779 samples and the high dimensionality introduced by one-hot encoding and text vectorization, we must reduce the number of features to avoid overfitting and improve model generalization
#%%
df[['country', 'location','title']]
#%%
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=20)
X_new = selector.fit_transform(X, y)
selected_cols = X.columns[selector.get_support()]
X_new = pd.DataFrame(X_new, columns=selected_cols, index=X.index)
X_new
#%%
X = X.loc[:, ~X.columns.str.startswith('location_')]
X = X.loc[:, ~X.columns.str.startswith('title_')]
X
#%% md
# ### Scaling our features
#%% md
# We must scale our features which are now all numerical to ensure they are on a comparable scale. This is important because features with larger ranges can dominate the learning process and lead to unstable model training and convergence issues.
# 
# If a feature is too large, its weight will dominate the output of its corresponding neuron, making it difficult for the model to learn from other features. Conversely, if a feature is too small, its weight will have little impact on the output, making it hard for the model to learn from that feature.
# 
# Furthermore, if the feature is large and dominates the output of the neuron, its weights cost function derivative (its gradient) will also be large, leading to larger weight updates during backpropagation. This can cause the model to overshoot the optimal weights and result in unstable training. On the other hand, if the feature is small, its weights cost function derivative will be small, leading to smaller weight updates during backpropagation. This can slow down the learning process and make it difficult for the model to converge to the optimal weights.
# 
# We can avoid these issues by scaling our features to a similar range, ensuring that no single feature dominates the learning process and allowing the model to learn from all features effectively without being influenced by their original scales.
# 
# We will be using StandardScalar, which is z-score normalization, to scale our features. This scales the features to have a mean of 0 and a standard deviation of 1.
#%%
#splitting our dataset
from sklearn.model_selection import train_test_split
date_time, X = X['date_time'], X.drop(columns=['date_time'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #This is z-score normalization
scaler = scaler.fit(X_train)
#%%
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.concat([date_time, pd.DataFrame(X_train)], axis=1)
X_test = pd.concat([date_time, pd.DataFrame(X_test)], axis=1)
#%%
X_train['date_time']
#%% md
# ### An imbalnced dataset
#%%
plt.figure(figsize=(6,6))
plt.pie(y.value_counts(), labels=['No Tsunami (0)', 'Tsunami (1)'], autopct='%1.1f%%')
plt.title('Tsunami vs Non-Tsunami Events')
plt.axis('equal')
plt.show()
#%% md
# This pie chart shows that our dataset is imbalanced, with a significantly higher proportion of non-tsunami events (0) compared to tsunami events (1). This imbalance can pose challenges for modeling, as models may become biased towards the majority class and struggle to accurately predict the minority class.
# 
# #### Class weights
# 
# In order to address this class imbalance during model training, we can use class weights. Class weights assign a higher weight to the minority class (tsunami events) and a lower weight to the majority class (non-tsunami events). This helps the model pay more attention to the minority class during training, improving its ability to correctly classify tsunami events.
# 
# When using class weights, the model will receive a higher cost for misclassifying tsunami events compared to non-tsunami events. This encourages the model to learn patterns that are more relevant for predicting tsunami occurrences, ultimately leading to better performance on the minority class.
# 
# #### Left Skewed Features
# 
# We can also see in our earlier visualizations that several features have left skewed distributions, shown in both figure 1 and figure 3. We are limited in how we can address this skewness due to the small dataset size of 779 samples. However, we have already applied log transformation to several features with extreme outliers which has helped to reduce skewness and make the distributions more compact.
# 
# This skewness may still impact model performance, especially for algorithms that assume normally distributed features, such as logistic regression. However, given our dataset size constraints, we will proceed with the current feature distributions.
# 
#%% md
# ### Data quality summary
# 
# Through out the EDA and data preprocessing steps, we have identified and addressed several data quality issues that could impact our modeling efforts:
# 
# - The dataset is generally small with only 779 samples after cleaning, which limits the complexity of models we can use without overfitting. We have taken care to avoid overfitting.
# 
#  - It seems our dataset has a left skewed distribution of across serveral features, which may impact model performance. However, given our dataset size constraints, we have applied log transformation to features with extreme outliers to help reduce skewness where possible.
# 
# - We identified and handled missing values in several features using appropriate imputation strategies based on the nature of the data and relationships between features. However, this is not a perfect solution, and some uncertainty remains due to the imputation process, but given the limitation created by the missing data and our dataset size, this is the best approach available.
# 
# - Our dataset is not only small but also from 2001 until 2023 (Chauhan, 2024), given the influence of climate change on natural disasters, there is a possibility that patterns in the data may have shifted after the data collection period. This introduces additional uncertainty, as the model may not fully capture current trends in earthquake and tsunami occurrences.
# 
# - USGS is our sole data source, which may introduce collection bias or limitations based on their data collection methods and coverage. Relying on a single source can limit the diversity of data and potentially overlook important factors influencing tsunami occurrences. This is confirmed by (United States Geological Survey, 2022) which acknowledges that not all earthquakes are recorded and minor earthquakes within the United States are more likely to be captured than those in other regions.
# 
# 
#%%
#TODO: Hyperparameter tuning (Grid and Randomized Search)
#%%
#TODO: explore auto parameter tuning
#%% md
# ### References:
# 
# Chauhan, C. (2025) Earthquake dataset. Kaggle. Available from: https://www.kaggle.com/datasets/warcoder/earthquake-dataset?select=earthquake_data.csv [Accessed 24 October 2025].
# 
# United States Geological Survey (USGS) (no date) Earthquake Hazards Program Glossary. U.S. Geological Survey. Available from: https://www.usgs.gov/glossary/earthquake-hazards-program [Accessed 26 October 2025].
# CDI Wald, D.J., Quitoriano, V., Worden, C.B., Hopper, M. and Dewey, J.W. (2011) ‘USGS “Did You Feel It?”: Internet-based macroseismic intensity maps’, Annals of Geophysics, 54(6), pp. 688–707. Available from: https://www.researchgate.net/publication/268368064_USGS_Did_You_Feel_It_Internet-based_macroseismic_intensity_maps [Accessed 26 October 2025].
# 
# United States Geological Survey (USGS) (1989) Modified Mercalli Intensity Scale. U.S. Geological Survey. Available from: https://www.usgs.gov/programs/earthquake-hazards/modified-mercalli-intensity-scale [Accessed 26 October 2025].
# United States Geological Survey (USGS) (2010) PAGER Scientific Background. U.S. Geological Survey. Available from: https://earthquake.usgs.gov/data/pager/background.php [Accessed 26 October 2025].
# 
# United States Geological Survey (no date) PAGER Scientific Background. U.S. Geological Survey. Available from: https://earthquake.usgs.gov/data/pager/background.php [Accessed 3 November 2025].
# 
# itdxer (2017) ‘What does average of word2vec vector mean?’, Cross Validated – Statistics Stack Exchange, 14 December. Available from: https://stats.stackexchange.com/questions/318882/what-does-average-of-word2vec-vector-mean[Accessed 3 November 2025].
# 
# United States Geological Survey (2022) Why is the earthquake that was reported/recorded by network X, or that I felt, not on the Latest Earthquakes map/list? Available from: https://www.usgs.gov/faqs/why-earthquake-was-reportedrecorded-network-x-or-i-felt-not-latest-earthquakes-maplist [Accessed 3 November 2025]
# 