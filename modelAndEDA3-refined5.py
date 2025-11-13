#%% md
# https://www.kaggle.com/datasets/warcoder/earthquake-dataset?select=earthquake_1995-2023.csv
# 
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
# When deciding if a variable is categorical or numerical we considered that:
# - Numerical variables typically have a large number of unique values (e.g., more than 15-20 unique values).
# - Categorical variables usually have a limited number of unique values (e.g., less than 15-20 unique values) and often represent distinct categories or groups.
# - Categorical variables can also be identified by their data type (e.g., 'object' in pandas).
# - Textual data can be identified by sampling the data and looking at the values.
# 
#  According to the USGS, `mmi` and `cdi` are both intensity scales that measure the effects of an earthquake, while `magnitude` measures the energy released. `magnitude` will therefore be treated numerical as it represents a continuous scale of energy release but `cdi` and `mmi` will be treated as ordered categorical features as they represent discrete intensity levels.
# (United States Geological Survey, no date; Wald et al., 2011; United States Geological Survey, 1989)
# 
# 
#  `title` and `location` are unstructured text features because they are of object data type but have a higher relative unique value counts (shown in bar chart) of more free-form text data. `date_time` is a datetime feature.  Our target variable `tsunami` is a binary numerical variable having only has 2 unique values (0 and 1).
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
# Observing (Chauhan, 2025, About Dataset), the following features stand out as particularly relevant for predicting tsunamis :
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
# 
# 
# The dataset has a left skewed distribution of earthquake magnitudes and impacts (figure 1), hence the lower severity alert levels (green and yellow) are more common than higher severity levels (orange and red). This strongly suggests that alert levels are related to earthquake characteristics which we can leverage for imputation.
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

#Order of alert levels: red, orange, yellow, green - source: (United States Geological Survey, no date, section “PAGER Scientific Background”)

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
df['alert'] = df.apply(impute_alert, axis=1)
print(f'Missing values in alert column after imputation: {df["alert"].isnull().sum()}')
#%% md
# #### Location missing values imputation strategy
#%%
msno.matrix(df)
#%% md
# `Location` is closely related to the `country` feature as it describes the area within the country. `location` is an unstructured text feature and `country` categorical text feature and, given they are the same type of data, we can use one to impute missing values in the other.
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
# The random samples above show many of the location entries contain a comma, indicating both a specific area and the country name (e.g., "Los Angeles, USA"). This information can help us populate missing country values from the location column.
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
# Many of the 42 samples without country data are small islands, territories, ocean regions, or disputed areas. I'm confident these values represent similar value to countries and will therefore impute these missing country values with the location value itself for these instances.
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
# Data missing from the `contient` column is 73%. Given this high level of missingness, it is therefore more practical to drop the 'continent' column entirely from the dataset rather than attempting to impute the missing values. Imputing such a large proportion of missing data could introduce significant bias and uncertainty into the dataset.
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
# ### Encoding Categorical Features and Text Features
#%% md
# We must encode categorical features into numerical format before modeling as most machine learning algorithms require numerical input. We will use one-hot encoding for nominal categorical features and ordinal encoding for ordinal categorical features (figure 3). Not all models require this purely numerical input, such as decision trees, but to ensure a fair comparison between different model types later on we will convert all categorical features to numerical format now.
# 
# `alert`, is a ordinal categorical variable so we will apply ordinal encoding as it is currently in a categorical text format and needs to be converted to numerical format for modeling.
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
# Having imputed missing values in the `country` feature earlier its unique value count has grown from 49 to 87 unique values. This is expected as we filled in missing country values using location data, which introduced new country names that were not present before imputation.
# 
# As country is a nominal categorical variable, we would need to hot-one encode it, this would create 87 new columns. This will greatly increase the dimensions in our dataset, likely creating overfitting. We are also aware that 91.4% of location values likely contain country information anyway, so I will drop this feature.
#%%
X = X.drop(columns=['country'])
X
#%% md
# Now we must process our unstructured text features: `title` and `location` (figure 3). These features contain free-form text data that cannot be directly used in most ML models so we will need to convert this text data into numerical format using text vectorization techniques.
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
# Averaging vectors Inspired by (itdxer, 2017)
#%%
object_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f'Object type columns in X: {object_columns}')
#%%
X['time'] = pd.DataFrame({'time': pd.to_datetime(X['date_time'])})
X['time'] = pd.to_datetime(X['time']).astype(int)/ 10**9 #unix timestamp
X = X.drop(columns=['date_time'])
#%% md
# (AlwaysSunny, 2019)
#%%
object_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f'Object type columns in X: {object_columns}')
datatime_columns = X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
print(f'Datetime type columns in X: {datatime_columns}')
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
# ### Scaling our features
#%% md
# We must scale our features which are now all numerical to ensure they are on a comparable scale. This is important because features with larger ranges can dominate the learning process and lead to unstable model training and convergence issues.
# 
# We can avoid these issues by scaling our features to a similar range, ensuring that no single feature dominates the learning process and allowing the model to learn from all features effectively without being influenced by their original scales.
# 
# We will be using StandardScalar, which is z-score normalization, to scale our features. This scales the features to have a mean of 0 and a standard deviation of 1.
#%% md
# #### Data Splitting
# 
# Due to our small dataset size (779 samples), we will use an 80-20 train-test split to ensure we have enough data for both training and testing our models. We will also use stratified sampling to maintain the same proportion of tsunami and non-tsunami events in both the training and testing sets, which is important given the class imbalance in our target variable.
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #This is z-score normalization
scaler = scaler.fit(X_train)
#%%
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=X.columns, index=y_train.index)
X_test = pd.DataFrame(X_test, columns=X.columns, index=y_test.index)
#%% md
# ### Feature Reduction, analysis and selection
# 
#%%
df[['country', 'location','title']].sample(16)
#%% md
# The similarity 
#%%
X_train = X_train.loc[:, ~X_train.columns.str.startswith('title_')]
X_test = X_test.loc[:, ~X_test.columns.str.startswith('title_')]
X_train
#%%
from sklearn.ensemble import RandomForestClassifier
def calculate_gini_importance(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    importances = clf.feature_importances_
    return pd.DataFrame({'Features': X.columns, 'GINI Importance': importances}).sort_values('GINI Importance', ascending=False)

feature_imp_df = calculate_gini_importance(X_train, y_train)
#%%
plt.figure(figsize=(20, 15))
plt.barh(feature_imp_df['Features'], feature_imp_df['GINI Importance'], color='skyblue')
plt.xlabel('GINI Importance')
plt.title('Feature Importance GINI Importance')
plt.gca().invert_yaxis()
plt.show()
#%% md
# Implementation based on: (GeeksForGeeks, 2025)
#%%
#model is learning temporal patterns instead of earthquake patterns, so lets drop time.
X_train_importance = X_train.drop(columns=['time'])
X_train_importance = calculate_gini_importance(X_train_importance, y_train)
#%%
top_5_features = X_train_importance.head(5)
plt.figure(figsize=(10, 6))
plt.barh(top_5_features['Features'], top_5_features['GINI Importance'], color='lightcoral')
plt.xlabel('GINI Importance')
plt.title('Top 5 Feature Importance - GINI Importance')
plt.gca().invert_yaxis()
plt.show()
#%%
X_train = X_train.drop(columns=['time'])
X_test = X_test.drop(columns=['time'])
#%%
net_columns = []
for col in X_train.columns:
    if col.startswith('net_'):
        net_columns.append(col)
net_importance = X_train_importance[X_train_importance['Features'].isin(net_columns)]
print(net_importance)
#%%
X_train = X_train.drop(columns=net_columns)
X_test = X_test.drop(columns=net_columns)
#%%
location_vector_columns = [col for col in X_train.columns if col.startswith('location_vec_')]
location_vector_importance = X_train_importance[X_train_importance['Features'].isin(location_vector_columns)]
total_location_vector_importance = location_vector_importance['GINI Importance'].sum()
print(f'Total Gini Importance of location vector features: {total_location_vector_importance}')
#compare with nst importance
nst_importance = X_train_importance[X_train_importance['Features'] == 'nst']['GINI Importance'].values[0]
print(f'Gini Importance of nst feature: {nst_importance}')
#plot comparison
plt.bar(['Location Vectors', 'nst'], [total_location_vector_importance, nst_importance], color=['orange', 'blue'])
plt.title('Gini Importance: Location Vectors vs nst Feature')
plt.ylabel('Gini Importance')
plt.show()
#%% md
# Sections of this code were generated with (ChatGPT, 2025)
#%%
magtype_drop_columns = ['magType_mb', 'magType_ms', 'magType_ml', 'magType_md', 'magType_mw', 'magType_mwb']
X_train = X_train.drop(columns=magtype_drop_columns)
X_test = X_test.drop(columns=magtype_drop_columns)
#%%
#feature engineer a general significance feature
X_train['general_significance'] = X_train[['sig', 'magnitude', 'cdi', 'mmi']].mean(axis=1)
X_test['general_significance'] = X_test[['sig', 'magnitude', 'cdi', 'mmi']].mean(axis=1)
#%%
#General position feature
X_train['general_position'] = X_train[['latitude', 'longitude', 'depth']].mean(axis=1)
X_test['general_position'] = X_test[['latitude', 'longitude', 'depth']].mean(axis=1)
#%%
feature_imp_df = calculate_gini_importance(X_train, y_train)
plt.figure(figsize=(20, 15))
plt.barh(feature_imp_df['Features'], feature_imp_df['GINI Importance'], color='skyblue')
plt.xlabel('Gini Importance')
plt.title('Feature Importance - Gin Importance')
plt.gca().invert_yaxis()
plt.show()
#%%
from sklearn.inspection import permutation_importance
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
perm_imp_df = pd.DataFrame({'Feature': X_train.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
#%%
plt.figure(figsize=(20, 15))
plt.barh(perm_imp_df['Feature'], perm_imp_df['Permutation Importance'], color='lightgreen')
plt.xlabel('Permutation Importance')
plt.title('Feature Importance - Permutation Importance')
plt.gca().invert_yaxis()
plt.show()
#%%
import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train)
shap_values_class1 = shap_values[:, :, 1]
shap.summary_plot(shap_values_class1, X_train, plot_type='dot')
#%% md
# #### Feature Importance Ranking (Combined Influence) & Domain Relevance
# 
# 
# | **Rank** | **Feature Name**           | **Domain Relevance**                                                |
# |----------|----------------------------|---------------------------------------------------------------------|
# | 1        | Location Vectors (merged)  | Encodes geographic patterns and seismic hotspots.                   |
# | 2        | dmin                       | Distance to nearest station, affects location accuracy.             |
# | 3        | nst                        | Number of reporting stations, higher counts improve reliability.    |
# | 4        | magType_mww                | Moment magnitude method, affects magnitude measurement.             |
# | 5        | magType_mwc                | Alternatve magnitude method, aids in cross-checking readings.       |
# | 6        | longitude                  | Longitude coordinate, key for mapping and regional analysis.        |
# | 7        | cdi                        | Reported intensity, indicates how strongly the quake was felt.      |
# | 8        | latitude                   | Latitude coordinate, identifes location.                            |
# | 9        | general_significance       | Mean of `sig`, `magnitude`, `cdi`, and `mmi`;                       |
# | 10       | general_position           | Mean of `latitude`, `longitude`, and `depth`                        |
# | 11       | depth                      | Earthquake focus depth, shallower quakes cause more surface damage. |
# 
# SHAP and GINI interpretation combined with domain knowledge to rank feature importance.
# 
#%% md
# ### An imbalnced dataset
#%%
plt.figure(figsize=(6,6))
plt.pie(y.value_counts(), labels=['No Tsunami (0)', 'Tsunami (1)'], autopct='%1.1f%%')
plt.title('Tsunami vs Non-Tsunami Events')
plt.axis('equal')
plt.show()
#%% md
# This pie chart shows that our dataset is imbalanced, with a significantly higher proportion of non-tsunami events (0) compared to tsunami events (1)
# #### Class weights
# 
# In order to address this class imbalance during model training, we can use class weights. Class weights assign a higher weight to the minority class (tsunami events) and a lower weight to the majority class (non-tsunami events). This helps the model pay more attention to the minority class during training, improving its ability to correctly classify tsunami events.
# 
# SMOTE is another technique for handling imbalanced datasets by generating synthetic samples for the minority class. However, given our small dataset size and high dimensionality due to embedding features, using SMOTE could lead to overfitting and may not be effective. Therefore, we will use class weights to address the class imbalance in our dataset. (Blagus & Lusa,2013)
#%% md
# ### Data limitation and quality issues
# 
# Throughout the EDA and data preprocessing steps, we have identified and addressed several data quality issues that could impact our modeling efforts:
# 
# - The dataset is generally small with only 779 samples after cleaning, which limits the complexity of models we can use without overfitting. We have taken care to avoid overfitting.
# 
#  - It seems our dataset has a left skewed distribution of across serveral features, which may impact model performance. However, given our dataset size constraints, we have applied log transformation to features with extreme outliers to help reduce skewness where possible.
# 
# - We identified and handled missing values in several features using appropriate imputation strategies based on the nature of the data and relationships between features. However, this is not a perfect solution, and some uncertainty remains due to the imputation process, but given the limitation created by the missing data and our dataset size, this is the best approach available.
# 
# - USGS is our sole data source, which may introduce collection bias or limitations based on their data collection methods and coverage. Relying on a single source can limit the diversity of data and potentially overlook important factors influencing tsunami occurrences. This is confirmed by (United States Geological Survey, 2022) which acknowledges that not all earthquakes are recorded and minor earthquakes within the United States are more likely to be captured than those in other regions.
# 
# 
#%% md
# ### Ethics and Privacy Considerations
# 
# This dataset contains no personally identifiable information bout individuals. However, there are still important ethical considerations to keep in mind when working with this data. Especially around deployment, any model used for tsunami prediction must be rigorously tested to avoid false positives or negatives that could lead to unnecessary concern or risk among populations at risk. Furthermore, a consideration must be made for accessibility for poorer and remote communities who may not have access to expensive high compute resources.
# 
# To address these ethical considerations, we will ensure that our modeling process is transparent and well-documented, allowing for validation by experts in the field. We should also avoid complex resource intensive models that may not be deployable in low-resource settings, like remote communities. Finally, we will communicate the limitations of our model clearly, emphasizing that it is a tool to aid decision-making rather than a definitive predictor of tsunami events.
# 
# When developing the model we will focus on precision and recall metrics to minimize false negatives (missed tsunami events) while also controlling false positives (incorrect tsunami warnings). This balanced approach will help ensure that the model is both effective and responsible in its predictions. We will also choose algorithms that aren't too complex to deploy in low-resource settings, ensuring accessibility for all communities at risk.
#%% md
# ### Model Training
#%% md
# We will use cross validation to evaluate our models, specifically stratified k-fold cross-validation. This method ensures that each fold maintains the same proportion of tsunami and non-tsunami events as the overall dataset, which is important given the class imbalance in our target variable.
#%%
from sklearn.model_selection import StratifiedKFold, cross_val_score
k_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X_recombined, y_recombined = pd.concat([X_train, X_test], axis=0), pd.concat([y_train, y_test], axis=0)
#%% md
# #### Model 1: Logistic Regression with Class Weights
# 
# Logistic Regression is a simple yet effective algorithm for binary classification tasks like ours. It models the probability of the positive class (tsunami event) using a sigmoid function. When using Logistic regression, we can apply class weights to address the class imbalance in our dataset. Its also preforms well when you have large number of features compared to samples, which is the case with our dataset after text vectorization.
#%%
#logistic regression with class weights and loss curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logclf = LogisticRegression(class_weight='balanced', max_iter=1000)
logclf.fit(X_train, y_train)
#%%
predictions_clf = logclf.predict(X_test)
logclf_cross_val_scores = cross_val_score(logclf, X_recombined, y_recombined, cv=k_folds)
print(f'Logistic Regression Cross-Validation Scores: {logclf_cross_val_scores}')
print(f'Logistic Regression Mean CV Score: {logclf_cross_val_scores.mean()}')
#%%
#TODO: Hyerparameter tuning
#%% md
# #### Model 2: Random Forest Classifier with Class Weights
#%% md
# Gradient Boosting Classifier is an ensemble learning method that builds multiple weak learners (decision trees) sequentially, where each tree tries to correct the errors of the previous ones. It combines the predictions of all trees to make a final prediction. Decision trees are not overall sensitive to outliers in its features, so this model should be robust to the extreme values in our dataset even after log transformation. Furthermore, we can still use our class weights to address the class imbalance during training.
# 
# Decsion trees are the 'standard' approach for tabular data and often outperform more complex models like neural networks on such datasets, especially when the dataset is small to medium sized like ours. There simplicity and low computational requirements also make them easier to deploy in low-resource settings, which is an important ethical consideration for our tsunami prediction model.
#%%
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
#%%
predictions_gbc = gbc.predict(X_test)
gbc_cross_val_scores = cross_val_score(gbc, X_recombined, y_recombined, cv=k_folds)
print(f'Gradient Boosting Classifier Cross-Validation Scores: {gbc_cross_val_scores}')
print(f'Gradient Boosting Classifier Mean CV Score: {gbc_cross_val_scores.mean()}')
#%%
#TODO: Hyperparameter tuning
#%%
#TODO: Hyperparameter tuning (Grid and Randomized Search)
#%%
#TODO: explore auto parameter tuning
#%%
#TODO: NOVEL - Bag both models together and see if it improves performance - does confusion matrix hint at this?
#%% md
# ### References:
# 
# Chauhan, C. (2025) Earthquake dataset. Kaggle. Available from: https://www.kaggle.com/datasets/warcoder/earthquake-dataset?select=earthquake_data.csv [Accessed 24 October 2025].
# 
# United States Geological Survey (USGS) (no date) Earthquake Hazards Program Glossary. U.S. Geological Survey. Available from: https://www.usgs.gov/glossary/earthquake-hazards-program [Accessed 26 October 2025].
# 
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
# A l w a y s S u n n y, 2019. pandas convert from datetime to integer timestamp [online]. Stack Overflow. Available at: https://stackoverflow.com/questions/54312802/pandas-convert-from-datetime-to-integer-timestamp [Accessed 4 Nov. 2025].
# 
# GeeksforGeeks, 2025. Feature Importance with Random Forests. [online] Available at: https://www.geeksforgeeks.org/machine-learning/feature-importance-with-random-forests/?utm_source=chatgpt.com [Accessed 4 November 2025].
# 
# ChatGPT, 2025. General multi-use session. [online] Available at: https://chatgpt.com/c/6909a310-4050-832b-94b1-35a82b81cec3 [Accessed 4 November 2025].
# 
# Blagus, R. & Lusa, L., 2013. SMOTE for high-dimensional class-imbalanced data. BMC Bioinformatics, 14, p.106.
# Available at: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-106 [Accessed 5 November 2025].