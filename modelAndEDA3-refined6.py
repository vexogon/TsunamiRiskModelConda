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
# Will a gradient decsion tree model outperform logistic regression in predicting tsunami occurrence following earthquakes based on seismic event characteristics?
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
from sklearn.model_selection import train_test_split, LearningCurveDisplay

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
#TODO: Explain why (similarity between entries)
#%% md
# 
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
# This dataset contains no personally identifiable information bout individuals. However, there are still important ethical considerations to keep in mind when working with this data. Especially around deployment, any model used for tsunami prediction must be rigorously tested to avoid false positives or negatives that could lead to unnecessary concern or danger for populations at risk. Furthermore, a consideration must be made for accessibility for poorer and remote communities who may not have access to expensive high compute resources.
# 
# To address these ethical considerations, we will ensure that our modeling process is transparent and well-documented, allowing for validation by experts in the field. We should also avoid complex resource intensive models that may not be deployable in low-resource settings, like remote communities. Finally, we will communicate the limitations of our model clearly, emphasizing that it is a tool to aid decision-making rather than a definitive predictor of tsunami events.
# 
# When developing the model we will focus on precision and recall metrics to minimize false negatives (missed tsunami events) while also controlling false positives (incorrect tsunami warnings). This balanced approach will help ensure that the model is both effective and responsible in its predictions. We will also choose algorithms that aren't too complex to deploy in low-resource settings, ensuring accessibility for all communities at risk.
#%% md
# ## Model Training
#%% md
# We will use cross validation to evaluate our models, specifically stratified k-fold cross-validation. This method ensures that each fold maintains the same proportion of tsunami and non-tsunami events as the overall dataset, which is important given the class imbalance in our target variable.
#%%
from sklearn.model_selection import StratifiedKFold, cross_val_score
k_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X_recombined, y_recombined = pd.concat([X_train, X_test], axis=0), pd.concat([y_train, y_test], axis=0)
#%%
def overfit_underfit_analysis(train_score, cross_val_score):
    print(f'Training Score: {train_score}')
    print(f'Cross Validation Score: {cross_val_score}')
    if train_score - cross_val_score > 0.01:
        print('The model is overfitting.')
        print(f'Difference: {train_score - cross_val_score}')
    elif cross_val_score - train_score > 0.01:
        print('The model is underfitting.')
        print(f'Difference: {cross_val_score - train_score}')
    else:
        print('The model has a good fit.')
#%% md
# ### Model 1: Logistic Regression with Class Weights
# 
# Logistic Regression is a simple yet effective algorithm for binary classification tasks like ours. It models the probability of the positive class (tsunami event) using a sigmoid function. When using Logistic regression, we can apply class weights to address the class imbalance in our dataset. Its also preforms well when you have large number of features compared to samples, which is the case with our dataset after text vectorization.
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logclf = LogisticRegression(class_weight='balanced')
logclf.fit(X_train, y_train)
#%%
logclf_train_score = accuracy_score(y_train, logclf.predict(X_train))
print(f'Accuracy Score on Training Set: {logclf_train_score}')
#%%
print(f'Accuracy Score on Test Set: {accuracy_score(y_test, logclf.predict(X_test))}')
#%%
predictions_clf = logclf.predict(X_test)
logclf_cross_val_scores = cross_val_score(logclf, X_recombined, y_recombined, cv=k_folds)
print(f'Logistic Regression Cross-Validation Scores: {logclf_cross_val_scores}')
print(f'Logistic Regression Mean CV Score: {logclf_cross_val_scores.mean()}')
#%%
overfit_underfit_analysis(logclf_train_score, logclf_cross_val_scores.mean())
#%% md
# #### Hyperparameter Tuning for Logistic Regression
# 
#  - Our analysis indicates slight overfitting, this is shown by the higher training score compared to cross-validation score. To address this, we can explore hyperparameter tuning to improve model generalization. This will likely adjustments to regularization.
# 
#  - Despite our high training score, we should still explore hyperparameter tuning to improve model performance on both the training and cross-validation sets. This can help us find the optimal settings for our model and potentially boost its accuracy further. This will likely require us to review a range of hyperparameters including regularization strength, solver choice, and maximum iterations.
# 
# - Usage of class weights. Although we have already implemented class weights to address class imbalance, we can experiment with different weight settings to see if they yield better performance. This could involve adjusting the weight ratio between classes or exploring alternative methods for handling imbalance.
# 
# 
# #### Regularization hyperparameter tuning (overfitting)
# 
# By adjusting the regularization strength (C) and type (penalty), we can help the model generalize better to unseen data. Regularization helps prevent overfitting by adding a penalty to the loss function for large weights. We will test different values of C to find the optimal balance between bias and variance. We will also test L1 and L2 regularization to see which works best for our dataset, as shrinking the models weights to zero with L1 can help with feature selection and reduce overfitting.
# 
#%%
hyperparameters = {
    'penalty':'l2', #regularization type
    'C':0.1, #regularization strength (inverse ratio - smaller values specify stronger regularization)
    'class_weight':'balanced' #address class imbalance (as before)
}

# Previous hyperparameter attempts:
#hyperparameters = {
#    'penalty':'l2',
#    'C':0.1,
#    'class_weight':'balanced'
#}

#hyperparameters = {
#    'penalty':'l2',
#    'C':0.05,
#    'class_weight':'balanced'
#}

#hyperparameters = {
#    'penalty':'l2',
#    'C':0.5,
#    'class_weight':'balanced'
#}

#hyperparameters = {
#    'penalty':'l1',
#    'solver':'saga',
#    'C':0.1,
#    'class_weight':'balanced'
#}
#%% md
# ***The regularization tuning above with C=0.01 yielded the best improvement in cross-validation score compared to previous attempts. Therefore, we will proceed with these hyperparameters for further evaluation and comparison with other models.***
#%%
logclf_tuned = LogisticRegression(**hyperparameters)
logclf_tuned.fit(X_train, y_train)
#%%
logclf_tuned_train_score = accuracy_score(y_train, logclf_tuned.predict(X_train))
print(f'Accuracy Score on Training Set: {logclf_tuned_train_score}')
logclf_tuned_test_score = accuracy_score(y_test, logclf_tuned.predict(X_test))
print(f'Accuracy Score on Test Set: {logclf_tuned_test_score}')
#%%
logclf_tuned_cross_val_scores = cross_val_score(logclf_tuned, X_recombined, y_recombined, cv=k_folds)
print(f'Logistic Regression Tuned Cross-Validation Scores: {logclf_tuned_cross_val_scores}')
print(f'Logistic Regression Tuned Mean CV Score: {logclf_tuned_cross_val_scores.mean()}')
#%%
overfit_underfit_analysis(logclf_tuned_train_score, logclf_tuned_cross_val_scores.mean())
#%%
print(f'Previous Logistic Regression Mean CV Score: {logclf_cross_val_scores.mean()}')
print(f'Tuned Logistic Regression Mean CV Score: {logclf_tuned_cross_val_scores.mean()}')
print(f'Improvement in Mean CV Score after Tuning: {logclf_tuned_cross_val_scores.mean() - logclf_cross_val_scores.mean()}')
#%% md
# #### Further Hyperparameter Tuning for general performance improvement (max iterations and tolerance)
# 
# By adjusting the maximum number of iterations and tolerance for convergence, we can potentially improve the model's performance further. Increasing the maximum iterations allows the model more opportunities to converge to an optimal solution and adjusting the tolerance can help fine-tune the stopping criteria for training, helping the model train for the maximum amount of useful time, before the changes in the loss function output become negligible.
#%%
hyperparameters = {
    'C':hyperparameters['C'],
    'class_weight':'balanced',
    'max_iter':100, #increase max iterations to ensure convergence
    'tol':1e-4 #tolerance for stopping criteria (how much improvement is required to continue training)
}
#previously attempted:
#hyperparameters = {
#    'C':0.01, #keep previous best C value
#    'class_weight':'balanced',
#    'max_iter':200,
#    'tol':1e-3
#}

#hyperparameters = {
#    'C':0.01, #keep previous best C value
#    'class_weight':'balanced',
#    'max_iter':300,
#    'tol':1e-5
#}

#hyperparameters = {
#    'C':0.01, #keep previous best C value
#    'class_weight':'balanced',
#    'max_iter':300,
#    'tol':1e-5
#}

#hyperparameters = {
#    'C':0.01, #keep previous best C value
#    'class_weight':'balanced',
#    'max_iter':500,
#    'tol':1e-4
#}   'solver':'liblinear' #Known to work well with small datasets

#%% md
# ***Various combinations of max iterations and tolerance were tested, but they did not yield significant improvements in cross-validation score compared to our previous regularization tuning.***
#%%
logclf_tuned_v2 = LogisticRegression(**hyperparameters)
logclf_tuned_v2.fit(X_train, y_train)
#%%
logclf_tuned_cross_val_scores_general_performance = cross_val_score(logclf_tuned_v2, X_recombined, y_recombined, cv=k_folds)
print(f'Logistic Regression Tuned Cross-Validation Scores: {logclf_tuned_cross_val_scores_general_performance}')
print(f'Logistic Regression Tuned Mean CV Score: {logclf_tuned_cross_val_scores_general_performance.mean()}')
#%%
overfit_underfit_analysis(logclf_tuned_train_score, logclf_tuned_cross_val_scores.mean())
#%%
print(f'Previous Logistic Regression Tuned Mean CV Score: {logclf_tuned_cross_val_scores.mean()}')
print(f'New Tuned Logistic Regression Mean CV Score: {logclf_tuned_cross_val_scores_general_performance.mean()}')
print(f'Improvement in Mean CV Score after Further Tuning: {logclf_tuned_cross_val_scores_general_performance.mean() - logclf_tuned_cross_val_scores.mean()}')
#%% md
# Tuning max iterations and tolerance did not yield significant improvements in cross-validation score compared to our previous regularization tuning. ***Therefore, we will proceed with the hyperparameters from our initial regularization tuning for further evaluation and comparison with other models.***
#%% md
# #### Class weight tuning
# 
# We will experiment with different class weight settings to see if they yield better performance. Although, removing class weights entirely is not advisable given the class imbalance in our dataset, we will still test this scenario to confirm its impact on model performance.
#%%
hyperparameters = {
    'C':0.01, #keep previous best C value
    'penalty':'l2',
    'class_weight':None, #remove class weights
}

#previously attempted:

# hyperparameters = {
#     'C':0.01, #keep previous best C value
#     'penalty':'l2',
#     'class_weight': {0: 2.0, 1: 1.0}, #adjust class weights to give more importance to non-tsunami class
# }
#%%
logclf_tuned_no_class_weights = LogisticRegression(**hyperparameters)
logclf_tuned_no_class_weights.fit(X_train, y_train)
#%%
logclf_tuned_no_class_weights_train_score = accuracy_score(y_train, logclf_tuned_no_class_weights.predict(X_train))
print(f'Accuracy Score on Training Set without Class Weights: {logclf_tuned_no_class_weights_train_score}')
logclf_tuned_no_class_weights_test_score = accuracy_score(y_test, logclf_tuned_no_class_weights.predict(X_test))
print(f'Accuracy Score on Test Set without Class Weights: {logclf_tuned_no_class_weights_test_score}')
#%%
logclf_tuned_no_class_weights_cross_val_scores = cross_val_score(logclf_tuned_no_class_weights, X_recombined, y_recombined, cv=k_folds)
print(f'Logistic Regression Tuned without Class Weights Cross-Validation Scores: {logclf_tuned_no_class_weights_cross_val_scores}')
print(f'Logistic Regression Tuned without Class Weights Mean CV Score: {logclf_tuned_no_class_weights_cross_val_scores.mean()}')
#%%
print(f'Previous Best Tuned Logistic Regression Mean CV Score: {logclf_tuned_cross_val_scores.mean()}')
print(f'Logistic Regression without Class Weights Mean CV Score: {logclf_tuned_no_class_weights_cross_val_scores.mean()}')
print(f'Change in Mean CV Score after Removing Class Weights: {logclf_tuned_no_class_weights_cross_val_scores.mean() - logclf_tuned_cross_val_scores.mean()}')
#%% md
# #### Grid Search for Hyperparameter Optimization
# 
# Given the insights from our previous hyperparameter tuning attempts, we will now perform a grid search to systematically explore a range of hyperparameter values for our Logistic Regression model. It appears that adjusting the regularization strength (C) could further enhance model performance. While max iterations and tolerance did not yield significant improvements previously, we will still include them in our grid search to ensure we find the optimal combination of hyperparameters.
# 
# The same applies to penalty type and solver choice, which we will also include in our grid search to identify the best settings for our dataset. Although previous attempts did not show significant improvements, exploring these hyperparameters in combination with others may lead to better overall performance.
# 
# We won't be including class weights in our grid search because even if a unique combination of other hyperparameters yields a slightly better score without class weights, it would not be advisable to use that model given the class imbalance in our dataset. Therefore, we will keep class weights fixed to 'balanced' during our grid search.
# 
#%% md
# | Hyperparameter    | What it does                                                                                                                                                                                         |
# |------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | `C`              | oppersite of regularization strength. Smaller values, stronger regularization, simpler model, less overfiting. Larger values, weaker regularization, more complex model, higher risk of overfitting. |
# | `penalty`        | Type of regularization applied. `'l2'` adds a penalty on squared weights to shrink them and reduce overfitting. `'l1'` (not in your grid) can create sparse coefficients.                            |
# | `class_weight`   | Adjusts the importance of each class to handle imbalance. `'balanced'` automatially scales weights inversely to class frequency.                                                                     |
# | `max_iter`       | maximum number of iterations for solver convergence. Too low, may not converge, higher values allow full optimization but slightly slower training.                                                  |
# | `tol`            | Tolerance for stopping criteria. Smaller values, solver runs longer for precise convergence, can slightly overfit. Larger values, stops earlier, may underfit but trains faster.                     |
# | `solver`         | our choice model type used to optimize the loss. `'liblinear'`, good for small datasets and supports L1 & L2 regularization                                                                          |
# 
#%%
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C':[0.005,0.01, 0.1, 0.3, 0.5],
    'penalty':['l2'], #keep l2 regularization
    'class_weight':['balanced'], #keep class weights to address class imbalance
    'max_iter':[100, 200, 300, 500], #vary max iterations
    'tol':[1e-2,1e-3, 1e-4, 1e-5], #vary tolerance
    'solver':['liblinear', 'lbfgs'] #try different solvers
}
grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=k_folds)
grid_search.fit(X_train, y_train)
grid_searched_best_model = grid_search.best_estimator_
grid_searched_logclf = grid_searched_best_model.fit(X_train, y_train)
#%%
print(f'Best Hyperparameters from Grid Search: {grid_search.best_params_}')
#%% md
# Our grid search is selecting within the region of our previous best hyperparameters, meaning its unlikely to yield significant improvements. However, we will still evaluate the grid searched model to confirm its status as our best model.
#%%
grid_searched_logclf_train_score = accuracy_score(y_train, grid_searched_logclf.predict(X_train))
print(f'Accuracy Score on Training Set: {grid_searched_logclf_train_score}')
grid_searched_logclf_test_score = accuracy_score(y_test, grid_searched_logclf.predict(X_test))
print(f'Accuracy Score on Test Set: {grid_searched_logclf_test_score}')
grid_searched_logclf_cross_val_scores = cross_val_score(grid_searched_logclf, X_recombined, y_recombined, cv=k_folds)
print(f'Grid Searched Logistic Regression Cross-Validation Scores: {grid_searched_logclf_cross_val_scores}')
print(f'Grid Searched Logistic Regression Mean CV Score: {grid_searched_logclf_cross_val_scores.mean()}')
#%%
#compare with previous best tuned model
print(f'Previous Best Tuned Logistic Regression Mean CV Score: {logclf_tuned_cross_val_scores.mean()}')
print(f'Grid Searched Logistic Regression Mean CV Score: {grid_searched_logclf_cross_val_scores.mean()}')
print(f'Improvement in Mean CV Score after Grid Search: {grid_searched_logclf_cross_val_scores.mean() - logclf_tuned_cross_val_scores.mean()}')
#%% md
# #### Logistic Regression hyperparameter tuning summary
#%%
import matplotlib.pyplot as plt

tuning_steps = ['Initial', 'Regularization Tuning', 'Further Tuning', 'Grid Search']
mean_cv_scores = [
    logclf_cross_val_scores.mean(),
    logclf_tuned_cross_val_scores.mean(),
    logclf_tuned_cross_val_scores_general_performance.mean(),
    grid_searched_logclf_cross_val_scores.mean()
]
plt.figure(figsize=(10,6))
plt.plot(tuning_steps, mean_cv_scores, marker='o', linestyle='-', color='purple', linewidth=2, markersize=8)

plt.title('Improvement in Mean CV Score after Each Tuning Step', fontsize=16)
plt.xlabel('Tuning Steps', fontsize=14)
plt.ylabel('Mean CV Score', fontsize=14)

# Zoom in on the range of scores to make small improvements more visible
plt.ylim(min(mean_cv_scores)-0.005, max(mean_cv_scores)+0.005) #(ChatGPT, 2025)

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%%
print("Logistic Regression Hyperparameter Tuning Summary:")
print(f'Initial Mean CV Score: {logclf_cross_val_scores.mean()}')
print(f'After Regularization Tuning Mean CV Score: {logclf_tuned_cross_val_scores.mean()}')
print(f'After Further Tuning Mean CV Score: {logclf_tuned_cross_val_scores_general_performance.mean()}')
print(f'After Grid Search Mean CV Score: {grid_searched_logclf_cross_val_scores.mean()}')
overfit_underfit_analysis(grid_searched_logclf_train_score, grid_searched_logclf_cross_val_scores.mean())
#%%
logclf_top_perf = grid_searched_logclf #top preforming model after hyperparameter tuning (identical to previous tuned model)
#%%
with open('logistic_regression_best_cv_score.txt', 'w') as f:
    f.write(f'Best Logistic Regression Mean CV Score: {grid_searched_logclf_cross_val_scores.mean()}')
    for i, score in enumerate(grid_searched_logclf_cross_val_scores):
        f.write(f'\nFold {i+1} CV Score: {score}')
    f.close()
#%% md
# #### Model 2: Gradient Boosting Classifier
#%% md
# Gradient Boosting Classifier is an ensemble learning method that builds multiple weak learners (decision trees) sequentially, where each tree tries to correct the errors of the previous ones. It combines the predictions of all trees to make a final prediction. Decision trees are not overall sensitive to outliers in its features, so this model should be robust to the extreme values in our dataset even after log transformation. Furthermore, we can still use our class weights to address the class imbalance during training.
# 
# Decsion trees are the 'standard' approach for tabular data and often outperform more complex models like neural networks on such datasets, especially when the dataset is small to medium sized like ours. There simplicity and low computational requirements also make them easier to deploy in low-resource settings, which is an important ethical consideration for our tsunami prediction model.
#%%
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
#%%
gbc_train_score = gbc.score(X_train, y_train)
print(f'Accuracy Score on Training Set: {gbc_train_score}')
#%%
print(f'Accuracy Score on Test Set: {accuracy_score(y_test, gbc.predict(X_test))}')
#%%
predictions_gbc = gbc.predict(X_test)
gbc_cross_val_scores = cross_val_score(gbc, X_recombined, y_recombined, cv=k_folds)
print(f'Gradient Boosting Classifier Cross-Validation Scores: {gbc_cross_val_scores}')
print(f'Gradient Boosting Classifier Mean CV Score: {gbc_cross_val_scores.mean()}')
#%%
overfit_underfit_analysis(gbc_train_score, gbc_cross_val_scores.mean())
#%% md
# ### Hyperparameter Tuning for Gradient Boosting Classifier
# 
# - We appear to have some reasonably significant overfitting, as indicated by the higher training score compared to cross-validation score. To address this, we can explore hyperparameter tuning to improve model generalization. This will likely involve adjustments to tree depth and number of estimators (number of trees in our ensemble).
# 
# - It be hard to improve on a 98% training score, so this indicates that we should focus our hyperparameter tuning efforts on reducing overfitting rather than boosting performance further.
#%% md
# #### Tree depth and number of estimators tuning (overfitting)
#%% md
# One of the main things that creates overfitting in decision tree based models is allowing the trees to grow too deep, which causes them to memorize the training data instead of learning general patterns. By limiting the maximum depth of the trees, we can help the model generalize better to unseen data. Additionally, adjusting the number of estimators (trees) in the ensemble can also impact overfitting. Too many trees can lead to overfitting. By finding the optimal number of estimators, we can balance bias and variance in our model.
#%%
hyperparameters = {
    'max_depth': 6, #limit tree depth to reduce overfitting
    'n_estimators': 150,#reduce number of trees to prevent overfitting
}

# Previous hyperparameter attempts:
# hyperparameters = {
#     'max_depth': 5,
#     'n_estimators': 200,
# }
# hyperparameters = {
#     'max_depth': 4,
#     'n_estimators': 100,
# }
# hyperparameters = {
#     'max_depth': 3,
#     'n_estimators': 200,
# }
# hyperparameters = {
#     'max_depth': 7,
#     'n_estimators': 200,
# }
#%%
gbc_tuned = GradientBoostingClassifier(**hyperparameters)
gbc_tuned.fit(X_train, y_train)
#%%
gbc_tuned_train_score = gbc_tuned.score(X_train, y_train)
print(f'Accuracy Score on Training Set: {gbc_tuned_train_score}')
gbc_tuned_test_score = accuracy_score(y_test, gbc_tuned.predict(X_test))
print(f'Accuracy Score on Test Set: {gbc_tuned_test_score}')
#%%
gbc_tuned_cross_val_scores = cross_val_score(gbc_tuned, X_recombined, y_recombined, cv=k_folds)
print(f'Gradient Boosting Classifier Tuned Cross-Validation Scores: {gbc_tuned_cross_val_scores}')
print(f'Gradient Boosting Classifier Tuned Mean CV Score: {gbc_tuned_cross_val_scores.mean()}')
#%%
overfit_underfit_analysis(gbc_tuned_train_score, gbc_tuned_cross_val_scores.mean())
#%%
print(f'Previous Gradient Boosting Classifier Mean CV Score: {gbc_cross_val_scores.mean()}')
print(f'Tuned Gradient Boosting Classifier Mean CV Score: {gbc_tuned_cross_val_scores.mean()}')
print(f'Improvement in Mean CV Score after Tuning: {gbc_tuned_cross_val_scores.mean() - gbc_cross_val_scores.mean()}')
#%% md
# #### Grid Search for Hyperparameter Optimization
# 
# Given the number of parameters we can tune from the Gradient boosting Classifier, we are going to preform a grid search. Given the insight we gained from inital testing about our overfitting our grid search will focus on related hyperparameters, such as:
# 
# | Hyperparameter           | What it does                                                                                                                    |
# |--------------------------|---------------------------------------------------------------------------------------------------------------------------------|
# | `max_depth`              | Limits the maximum depth of each tree. Deeper trees capture more complex patterns but are more likely to overfit.               |
# | `max_leaf_nodes`         | sets the maximum number of leaf nodes per tree. More leaves allow finer patterns, increasing overfitting risk.                  |
# | `min_samples_leaf`       | Minimum number of samples required at a leaf node. Higher values prevent very small leaves, reducing overfitting.               |
# | `min_samples_split`      | Minimum number of samples required to split a node. Larger values prevent splits on tiny nodes, reducing overfitting.           |
# | `learning_rate`          | Shrinks contribution of each tree in the ensemble. Smaller values slow learning but reduce overfitting.                         |
# | `n_estimators`           | Number of boosting iterations (trees). More trees can increase complexity and overfitting, especially with high learning rates. |
# 
# We will use the knowledge gained from our previous hyperparameter tuning attempts to guide our grid search, focusing on ranges that are likely to reduce overfitting while still allowing the model to learn effectively from the data.
#%%
#grid search
param_grid = {
    'max_depth':[5,6,7,8],
    'n_estimators':[100,150,200],
    'min_samples_split':[4,8],
    'min_samples_leaf':[2,4],
    'max_leaf_nodes':[20,30,40],
    'learning_rate':[0.01,0.05,0.1]
}
grid_search_gbc = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=k_folds)
grid_search_gbc.fit(X_train, y_train)
grid_searched_best_model_gbc = grid_search_gbc.best_estimator_
#%%
print(f'Best Hyperparameters from Grid Search: {grid_search_gbc.best_params_}')
grid_search_gbc_train_score = gbc_tuned.score(X_train, y_train)
print(f'Accuracy Score on Training Set: {grid_search_gbc_train_score}')
grid_search_gbc_test_score = accuracy_score(y_test, grid_searched_best_model_gbc.predict(X_test))
print(f'Accuracy Score on Test Set: {grid_search_gbc_test_score}')
grid_searched_gbc_cross_val_scores = cross_val_score(grid_searched_best_model_gbc, X_recombined, y_recombined, cv=k_folds)
print(f'Grid Searched Gradient Boosting Classifier Cross-Validation Scores: {grid_searched_gbc_cross_val_scores}')
print(f'Grid Searched Gradient Boosting Classifier Mean CV Score: {grid_searched_gbc_cross_val_scores.mean()}')
overfit_underfit_analysis(grid_search_gbc_train_score, grid_searched_gbc_cross_val_scores.mean())
#%%
import matplotlib.pyplot as plt

tuning_steps = ['Initial', 'Regularization Tuning', 'Grid Search']
mean_cv_scores = [
    gbc_cross_val_scores.mean(),
    gbc_tuned_cross_val_scores.mean(),
    grid_searched_gbc_cross_val_scores.mean()
]
plt.figure(figsize=(10,6))
plt.plot(tuning_steps, mean_cv_scores, marker='o', linestyle='-', color='purple', linewidth=2, markersize=8)

plt.title('Improvement in Mean CV Score after Each Tuning Step', fontsize=16)
plt.xlabel('Tuning Steps', fontsize=14)
plt.ylabel('Mean CV Score', fontsize=14)

# Zoom in on the range of scores to make small improvements more visible
plt.ylim(min(mean_cv_scores)-0.005, max(mean_cv_scores)+0.005) #(ChatGPT, 2025)

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%%
print(f'Previous Best Tuned Gradient Boosting Classifier Mean CV Score: {gbc_tuned_cross_val_scores.mean()}')
print(f'Grid Searched Gradient Boosting Classifier Mean CV Score: {grid_searched_gbc_cross_val_scores.mean()}')
print(f'Improvement in Mean CV Score after Grid Search: {grid_searched_gbc_cross_val_scores.mean() - gbc_tuned_cross_val_scores.mean()}')
#%%
gbc_top_perf = grid_searched_best_model_gbc #top preforming model after hyperparameter tuning
#%%
with open('gradient_decsion_tree_scores.txt', 'w') as f:
    f.write(f'Best Gradient Boosting Classifier Mean CV Score: {grid_searched_gbc_cross_val_scores.mean()}')
    for i, score in enumerate(grid_searched_gbc_cross_val_scores):
        f.write(f'\nFold {i+1} CV Score: {score}')
    f.close()
#%% md
# ### Discussion and Evaluation of results
#%%
print('Results Summary for Top Performing Models:')
print('\n')
print(f"Logistic Regression Training Score: {grid_searched_logclf_train_score}, \n Test Score: {grid_searched_logclf_test_score}, \n Mean CV Score: {grid_searched_logclf_cross_val_scores.mean()}")
print('\n')
print(f"Gradient Boosting Classifier training Score: {grid_search_gbc_train_score}, \n Test Score: {grid_search_gbc_test_score}, \n Mean CV Score: {grid_searched_gbc_cross_val_scores.mean()}")
mean_cv_score_difference = grid_searched_logclf_cross_val_scores.mean() - grid_searched_gbc_cross_val_scores.mean()
#%%
if mean_cv_score_difference > 0:
    print(f'Logistic Regression outperforms Gradient Boosting Classifier by {mean_cv_score_difference} on cross validation.')
else:
    print(f'Gradient Boosting Classifier outperforms Logistic Regression by {-mean_cv_score_difference} om cross validation.')
#%%
print('Overfit/Underfit Analysis for Top Performing Models:')
print("Logistic Regression Overfit/Underfit Analysis:")
overfit_underfit_analysis(grid_searched_logclf_train_score, grid_searched_logclf_cross_val_scores.mean())
print ("\n")
print("Gradient Boosting Classifier Overfit/Underfit Analysis:")
overfit_underfit_analysis(grid_search_gbc_train_score, grid_searched_gbc_cross_val_scores.mean())
#%%
import numpy as np
labels = ['Training Score', 'Test Score', 'Mean CV Score']
logclf_scores = [grid_searched_logclf_train_score, grid_searched_logclf_test_score, grid_searched_logclf_cross_val_scores.mean()]
gbc_scores = [grid_search_gbc_train_score, grid_search_gbc_test_score, grid_searched_gbc_cross_val_scores.mean()]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots(figsize=(10,6))
bars1 = ax.bar(x - width/2, logclf_scores, width, label='Logistic Regression', color='blue')
bars2 = ax.bar(x + width/2, gbc_scores, width, label='Gradient Boosting Classifier', color='orange')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Comparison of Top Performing Models', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
# Attach a text label above each bar in *bars*, displaying its height.
def autolabel(bars):
    """Attach a text label above each bar in *bars*, displaying its height."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
autolabel(bars1)
autolabel(bars2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%% md
# Code above developed with assistance from (ChatGPT, 2025)
#%%
models = ['Logistic Regression', 'Gradient Boosting Classifier']
gaps = [
    grid_searched_logclf_train_score - grid_searched_logclf_cross_val_scores.mean(),
    grid_search_gbc_train_score - grid_searched_gbc_cross_val_scores.mean()
]
plt.figure(figsize=(8,6))
bars = plt.bar(models, gaps, color=['blue', 'orange'])
plt.title('Gap between Training and Cross-Validation Scores', fontsize=16)
plt.ylabel('Score Gap', fontsize=14)
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%% md
# #### Train, Test and Cross-Validation Results Discussion
#%% md
# The results indicate both models perform well.
# 
# Both models preformed well on there training sets with both above 90%% accuracy. Indiecating they have both learned the training data well. The Gradient boosting classifier achieved 100% accuracy on the training set compared with just over 90% for logistic regression. Showing that the gradient boosting classifier has learned the training data better.
# 
# On the test set both models achieved similar accuracy scores of just over 85%. Indicating that both models generalised well to unseen data. the gradient boosting classifer achieved a slightly higher test accuracy, so it might be marginally better at generalising to unseen data compared to logistic regression.
# 
# The cross validation scores for both models were also similar. The gradient boosting classifier achieved a slightly higher mean cross validation score compared to logistic regression, indicating it might generalise slightly better across different data splits and is less sensitive to the specific training data used.
# 
# The larger gap between the cross validation and training scores for the gradient boosting classifier indicates it is overfitting the training data slightly more than the logistic regression model. This suggests that while the gradient boosting classifier learns the training data very well, but struggle to translate that performance into unseen data predictions. In contrast, the smaller gap for logistic regression indicates it generalises better from training to unseen data, making it potentially more robust.
# 
# 
#%%
from sklearn.metrics import precision_score, recall_score
logclf_predictions = grid_searched_logclf.predict(X_test)
logclf_precision = precision_score(y_test, logclf_predictions)
logclf_recall = recall_score(y_test, logclf_predictions)
print(f'Logistic Regression Precision: {logclf_precision}')
print(f'Logistic Regression Recall: {logclf_recall}')
#%%
gbc_predictions = grid_searched_best_model_gbc.predict(X_test)
gbc_precision = precision_score(y_test, gbc_predictions)
gbc_recall = recall_score(y_test, gbc_predictions)
print(f'Gradient Boosting Classifier Precision: {gbc_precision}')
print(f'Gradient Boosting Classifier Recall: {gbc_recall}')
#%%
from sklearn.metrics import PrecisionRecallDisplay
logclf_disp = PrecisionRecallDisplay.from_estimator(grid_searched_logclf, X_test, y_test, name="Logistic Regression", color='blue')
gbc_disp = PrecisionRecallDisplay.from_estimator(grid_searched_best_model_gbc, X_test, y_test, name="Gradient Boosting Classifier", color='orange', ax=logclf_disp.ax_)
logclf_disp.ax_.set_title('Precision & Recall Curve Comparison')
plt.show()
#%%
import numpy as np
labels = ['Precision', 'Recall']
logclf_scores = [logclf_precision, logclf_recall]
gbc_scores = [gbc_precision, gbc_recall]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots(figsize=(10,6))
bars1 = ax.bar(x - width/2, logclf_scores, width, label='Logistic Regression', color='blue')
bars2 = ax.bar(x + width/2, gbc_scores, width, label='Gradient Boosting Classifier', color='orange')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Precision and Recall Comparison of Top Performing Models', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
# Attach a text label above each bar in *bars*, displaying its height.
def autolabel(bars):
    """Attach a text label above each bar in *bars*, displaying its height."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
autolabel(bars1)
autolabel(bars2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%% md
# Code above developed with assistance from (ChatGPT, 2025)
#%%
from sklearn.metrics import f1_score
logclf_f1 = f1_score(y_test, logclf_predictions)
gbc_f1 = f1_score(y_test, gbc_predictions)
print(f'Logistic Regression F1 Score: {logclf_f1}')
print(f'Gradient Boosting Classifier F1 Score: {gbc_f1}')
#%% md
# #### Precision and Recall Discussion
# 
# Precision measures the accuracy of positive predictions, while recall measures the ability to find all positive instances. In our tsunami prediction context, high recall is crucial to minimize false negatives (missed tsunamis), while high precision is important to reduce false positives (false alarms), which can lead to unnecessary panic and resource allocation. This was previously discussed in the ethical and privacy considerations section, and makes precision and recall particularly important metrics for evaluating our models.
# 
# In precision, both models performed similarly, with the gradient boosting classifier achieving a slightly higher precision score. This indicates that when the gradient boosting classifier predicts a tsunami, it is marginally more likely to be correct compared to logistic regression. This is critical for reducing false alarms in tsunami warnings.
# 
# In recall, Logistic regression slightly outperformed the gradient boosting classifier, achieving a higher recall score. This suggests that logistic regression is better at identifying actual tsunami events, which is critical for ensuring that tsunamis are not missed. Missing a tsunami could have severe consequences, so a higher recall is another important metric for our models evaluation.
# 
# The higher F1 score of the gradient boosting classifier indicates it has a better balance between precision and recall compared to logistic regression. Given the importance of reducing both false alarms (precision) and missed tsunamis (recall), the gradient boosting classifier's superior F1 score suggests it may be the more effective model overall for our tsunami prediction task.
# 
# 
#%%
from sklearn.metrics import RocCurveDisplay
logclf_roc_disp = RocCurveDisplay.from_estimator(grid_searched_logclf, X_test, y_test, name="Logistic Regression", color='blue')
gbc_roc_disp = RocCurveDisplay.from_estimator(grid_searched_best_model_gbc, X_test, y_test, name="Gradient Boosting Classifier", color='orange', ax=logclf_roc_disp.ax_)
logclf_roc_disp.ax_.set_title('ROC Curve Comparison')
plt.show()
#%%
from sklearn.metrics import roc_auc_score
logclf_auc_roc = roc_auc_score(y_test, logclf_predictions)
print(f'Logistic Regression AUC ROC: {logclf_auc_roc}')
gbc_auc_roc = roc_auc_score(y_test, gbc_predictions)
print(f'Gradient Boosting Classifier AUC ROC: {gbc_auc_roc}')
#%% md
# #### ROC Curve and AUC Discussion
# 
# Both models performed well in terms of ROC AUC, indicating they are effective at distinguishing between tsunami and non-tsunami events across various classification thresholds. Both have a high True Positive Rate (sensitivity) while keeping a low False Positive Rate (specificity).
# 
# The gradient boosting classifier achieved a slightly higher AUC ROC score compared to logistic regression. This suggests that the gradient boosting classifier has a marginally better ability to differentiate between tsunami and non-tsunami events across all possible classification thresholds. This is an important consideration for our tsunami prediction task, as it indicates the model's robustness across situations and its potential effectiveness in real-world situations.
# 
#%%
from sklearn.metrics import ConfusionMatrixDisplay
logclf_cm_disp = ConfusionMatrixDisplay.from_estimator(grid_searched_logclf, X_test, y_test, display_labels=['No Tsunami', 'Tsunami'], cmap='Blues')
logclf_cm_disp.ax_.set_title('Logistic Regression Confusion Matrix')
plt.show()
#%%
gbc_cm_disp = ConfusionMatrixDisplay.from_estimator(grid_searched_best_model_gbc, X_test, y_test, display_labels=['No Tsunami', 'Tsunami'], cmap='Blues')
gbc_cm_disp.ax_.set_title('Gradient Boosting Classifier Confusion Matrix')
plt.show()
#%% md
# #### Confusion Matrix resukts discussion
# 
# The confusion matrices for both models provide insights into their classification performance on the test set.
# 
# Both misclassified a similar number of tsunami events as non-tsunamis (false negatives). The same applies for non-tsunamis misclassified as tsunamis (false positives). This indicates that both models have similar challenges in accurately distinguishing between tsunami and non-tsunami events. This was always probable given their similar precision and recall scores.
# 
# In conclusion, the confusion matrices confirm that both models have comparable classification performance, with similar patterns of misclassification. Both models preform reasonably well in general with regard to minimizing false negatives and false positives, which is critical for effective tsunami prediction. There is possibly room for improvement in reducing misclassifications further, which could be explored in future work.
#%%
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(grid_searched_logclf_cross_val_scores, grid_searched_gbc_cross_val_scores)
print(f'T-statistic: {t_stat}, P-value: {p_value}')
if p_value < 0.05:
    print('The difference in model performance is statistically significant.')
else:
    print('The difference in model performance is not statistically significant.')
#%% md
# #### Statistical Significance Testing Discussion
# 
# The above paired t-test compares the cross-validation scores of the two models to determine if there is a statistically significant difference in their performance. This is important to assess whether the observed differences in mean CV scores are likely due to chance or reflect a true difference in model effectiveness. The P value is less than 0.05, indicating that the difference in performance between the two models is statistically significant. This tells us that one model consistently outperforms the other across different data splits, providing confidence in the observed performance differences.
# 
# The large negative t-statistic indicates that the gradient boosting classifier generally outperforms logistic regression across the cross-validation folds. This aligns with our earlier observations of the mean CV scores, where the gradient boosting classifier had a slightly higher average score.
#%% md
# ### Research questions
#%% md
# `Will a gradient decsion tree model outperform logistic regression in predicting tsunami occurrence following earthquakes based on seismic event characteristics?`
# 
# Based on our analysis, the gradient boosting classifier does outperform logistic regression in predicting tsunami occurrence following earthquakes. The gradient boosting classifier achieved a higher mean cross-validation score, indicating better generalization across different data splits. Additionally, it had a slightly higher test accuracy compared to logistic regression. The gradient boosting classifier also demonstrated a better balance between precision and recall, as evidenced by its higher F1 score. This suggests that it is more effective at correctly identifying tsunami events while minimizing false alarms. Something that is critical given our earlier ethical and practical considerations.
#%%
import pandas as pd
feature_importances = grid_searched_best_model_gbc.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df = importance_df[importance_df['Importance'] >= 0.01]
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='orange')
plt.xlabel('Importance', fontsize=14)
plt.title('Feature Importances from Gradient Boosting Classifier', fontsize=16)
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%%
fi = permutation_importance(grid_searched_best_model, X_test, y_test, n_repeats=10, random_state=42)
#plot top features for logistic regression
importance_df_logclf = pd.DataFrame({'Feature': X_test.columns, 'Importance': fi['importances_mean']})
importance_df_logclf = importance_df_logclf.sort_values(by='Importance', ascending=False)
importance_df_logclf = importance_df_logclf[importance_df_logclf['Importance'] >= 0.01]
plt.figure(figsize=(10,6))
plt.barh(importance_df_logclf['Feature'], importance_df_logclf['Importance'], color='blue')
plt.xlabel('Importance', fontsize=14)
plt.title('Feature Importances from Logistic Regression', fontsize=16)
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%% md
# 
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
# ChatGPT, 2025. General multi-use session. [online] Available at: https://chatgpt.com/c/6909a310-4050-832b-94b1-35a82b81cec3 [Accessed 7 November 2025].
# 
# Blagus, R. & Lusa, L., 2013. SMOTE for high-dimensional class-imbalanced data. BMC Bioinformatics, 14, p.106.
# Available at: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-106 [Accessed 5 November 2025].