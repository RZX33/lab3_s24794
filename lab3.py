import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def cleaning_data(path, m_th=0.5, num_strategy='mean', cat_strategy='most_frequent'):
    df = pd.read_csv(path)
    print(df.head())
    print(df.info())
    print(f'Rows with missing elements \n {df.isnull().sum()}')
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Rows with empty values in dataset")
    plt.show()
    missing_per_row = df.isnull().mean(axis=1)
    df_clean = df[missing_per_row <= m_th].copy()
    deleted_rows = df[missing_per_row > m_th]
    print(f'Deleted {deleted_rows.index.size} rows')
    num_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    cat_columns = df_clean.select_dtypes(include=['object']).columns
    missing_per_row = df_clean.isnull().sum(axis=1)
    filled_rows = df_clean[missing_per_row > 0]
    imp = SimpleImputer(missing_values=np.nan, strategy=num_strategy)
    df_clean[num_columns] = imp.fit_transform(df_clean[num_columns])
    imp = SimpleImputer(missing_values=np.nan, strategy=cat_strategy)
    df_clean[cat_columns] = imp.fit_transform(df_clean[cat_columns])
    print(f'Filled missing data in {filled_rows.index.size} rows')
    print(df_clean.describe())
    print(df_clean.select_dtypes(include=['object']).describe())
    df_clean[num_columns].hist(figsize=(10, 8), bins=20, edgecolor='black')
    plt.tight_layout()
    plt.show()
    for column in cat_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=column, data=df_clean)
        plt.title(f'Number of values for column {column}')
        plt.xticks(rotation=45)
        plt.show()
    encoder = LabelEncoder()
    for column in cat_columns:
        df_clean[column] = encoder.fit_transform(df_clean[column])
    scaler = StandardScaler()
    df_clean[num_columns] = scaler.fit_transform(df_clean[num_columns])
    df_numeric = df.select_dtypes(include=[float, int])
    corr = df_numeric.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()
    return df_clean

def model_training_and_eval(model, X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(model)
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R2: {r2}')
    cv_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f'Mean MSE: {cv_scores.mean()}')
    return model

def model_creation(df):
    X = df.drop('score', axis=1)
    y = df['score']
    model = LinearRegression()
    model = model_training_and_eval(model, X, y)
    best_model = model
    model = RandomForestRegressor(random_state=1337)
    model = model_training_and_eval(model, X, y)
    model.n_estimators = 200
    model.max_depth = 5
    model = model_training_and_eval(model, X, y)
    return best_model


df = cleaning_data('CollegeDistance.csv')
model_creation(df)
