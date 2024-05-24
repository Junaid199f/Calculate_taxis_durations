import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_parquet('yellow_tripdata_2024-01.parquet')

# Calculate trip duration in minutes
df['Duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60

# Convert 'PULocationID' and 'DOLocationID' to strings
df['PULocationID'] = df['PULocationID'].astype(str)
df['DOLocationID'] = df['DOLocationID'].astype(str)

# Initial filtering by duration
filtered_df = df[(df['Duration'] >= 1) & (df['Duration'] <= 60)]

# Additional filtering to remove outliers, assuming a simple range method
# Adjust these bounds as necessary based on your specific outlier definition
duration_q1 = filtered_df['Duration'].quantile(0.25)
duration_q3 = filtered_df['Duration'].quantile(0.75)
iqr = duration_q3 - duration_q1
upper_bound = duration_q3 + 1.5 * iqr
lower_bound = duration_q1 - 1.5 * iqr

final_filtered_df = filtered_df[(filtered_df['Duration'] >= lower_bound) & (filtered_df['Duration'] <= upper_bound)]

# Create feature matrix
records = final_filtered_df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
vec = DictVectorizer(sparse=False)
feature_matrix = vec.fit_transform(records)

# Extract response variable
y = final_filtered_df['Duration'].values

# Fit the linear regression model
model = LinearRegression()
model.fit(feature_matrix, y)

# Predict and calculate RMSE
predictions = model.predict(feature_matrix)
rmse = np.sqrt(mean_squared_error(y, predictions))

print(f"Dimensionality of feature matrix: {feature_matrix.shape[1]}")
print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"y shape: {y.shape}")
print(f"RMSE of the model on the training data: {rmse}")

# Assuming 'vec' and 'model' are already defined and trained as per previous steps

# Load the validation dataset
df_validation = pd.read_parquet('yellow_tripdata_2024-02.parquet')

# Prepare the validation dataset
# Calculate trip duration in minutes
df_validation['Duration'] = (
                                        df_validation.tpep_dropoff_datetime - df_validation.tpep_pickup_datetime).dt.total_seconds() / 60

# Convert 'PULocationID' and 'DOLocationID' to strings
df_validation['PULocationID'] = df_validation['PULocationID'].astype(str)
df_validation['DOLocationID'] = df_validation['DOLocationID'].astype(str)

# Filter the validation DataFrame similarly to how the training DataFrame was filtered
filtered_validation_df = df_validation[(df_validation['Duration'] >= 1) & (df_validation['Duration'] <= 60)]

# Additional filtering to remove outliers, based on the same criteria as training data
duration_q1_val = filtered_validation_df['Duration'].quantile(0.25)
duration_q3_val = filtered_validation_df['Duration'].quantile(0.75)
iqr_val = duration_q3_val - duration_q1_val
upper_bound_val = duration_q3_val + 1.5 * iqr_val
lower_bound_val = duration_q1_val - 1.5 * iqr_val

final_filtered_validation_df = filtered_validation_df[
    (filtered_validation_df['Duration'] >= lower_bound_val) &
    (filtered_validation_df['Duration'] <= upper_bound_val)
    ]

# Create feature matrix using the DictVectorizer fitted on the training data
records_validation = final_filtered_validation_df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
feature_matrix_validation = vec.transform(records_validation)

# Extract response variable
y_validation = final_filtered_validation_df['Duration'].values

# Predict on the validation data
predictions_validation = model.predict(feature_matrix_validation)

# Calculate the RMSE on the validation data
rmse_validation = np.sqrt(mean_squared_error(y_validation, predictions_validation))

print(f"RMSE of the model on the validation data: {rmse_validation}")
