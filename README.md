# Chicago_crime_analysis
### **1. Data Preparation**

The dataset used for crime time prediction contains information about various crimes in Chicago, with features such as location, crime type, time, and other contextual data. It consists of over 8 million rows and 80 columns, including features like "Block," "IUCR" (crime type code), latitude, longitude, district, community area, and time features like month, day of the week, and hour.

The data was prepared by encoding categorical variables, scaling continuous features, and splitting into training and validation sets.

- **Data Splitting**: The dataset was split into training and validation sets, with an 80:20 split.
- **Target Variable**: The target variable for this model is "Hour," representing the hour in which the crime occurred.
- **Features**: The feature set includes both static and time-varying features, such as the location and type of crime, along with temporal features like month and day.

### **2. Handling Missing Values**

Handling missing values was an important part of data preparation. Missing data was found in various columns, particularly in geographical information and categorical identifiers. The following approaches were used:

- **Location Features**: Missing values in latitude and longitude were analyzed as they could significantly impact the crime location prediction. Rows with invalid latitude and longitude values were filtered out.
- **Categorical Variables**: Missing values in categorical variables such as "Beat" or "Community Area" were imputed using statistical methods like mode imputation.
- **Feature Engineering**: Linear regression was considered for imputing some columns where geographical information was involved, providing more precise replacements for missing values.

### **3. Outliers**

Outlier detection was performed on latitude and longitude values to ensure that the crime locations fall within Chicago's geographical boundaries. Significant numbers of outliers were detected in the longitude values. These outliers were filtered, and the dataset was recalibrated, ensuring all coordinates align with known boundaries for the city of Chicago.

- **Impact of Longitude Outliers**: Analysis showed that longitude outliers could affect clustering and hotspot detection. Proper filtering was applied to retain valid data points.

### **4. Feature Engineering**

The feature engineering process involved several steps to enhance the predictive power of the model:

- **Categorical Encoding**: The "IUCR" column, representing crime type codes, was encoded using Label Encoding, which transformed the 416 unique values into numerical values.
- **Temporal Features**: Features such as "Hour" and "Day of the Week" were transformed into sine and cosine components to capture cyclical patterns. These transformations were included in the final dataset as "Hour_sin," "Hour_cos," etc.
- **Clustering Features**: New features representing crime clusters, such as "Geo Cluster" and "Primary Type Grouped," were added to help identify patterns across different geographical areas.
- **Scaling**: Numerical columns were scaled using `StandardScaler` to bring all features to a comparable range, improving the performance of machine learning models.

### **5. Exploratory Data Analysis (EDA)**

EDA was performed to understand the temporal and spatial distribution of crime data:

- **Crime Hotspots**: Heatmaps and clustering were used to identify areas in Chicago with higher crime rates, particularly focused on areas with frequent incidents during specific hours.
- **Crime Type vs. Time Analysis**: Scatter plots and box plots were used to visualize how crime types vary throughout the day. A significant number of crimes tend to occur during nighttime hours, especially violent crimes.
- **Temporal Analysis**: Time-series plots showed the fluctuation in crime occurrences across different hours of the day, providing insights into peak crime hours.

### **6. Data Modeling**

#### **Initial Model - Temporal Fusion Transformer (TFT)**

- **Model Type**: A Temporal Fusion Transformer (TFT) was initially implemented using PyTorch Forecasting to predict the time of crime. This model was chosen due to its capability to handle large temporal datasets with static, known, and unknown real features.
- **Challenges Faced**: The model required the use of a 3D input tensor with the dimensions representing batch size, sequence length, and feature length. Several issues were faced related to tensor dimensions and GPU utilization, and the model was ultimately computationally expensive for this dataset.

#### **Improved Model - XGBoost Regressor**

- **Model Type**: The final model selected for crime time prediction was an XGBoost Regressor due to its speed and efficiency, especially when tuned for GPU acceleration.
- **Model Configuration**:
  - `n_estimators=500`: Number of boosting rounds.
  - `learning_rate=0.05`: Learning rate for model training.
  - `max_depth=6`: Depth of each tree.
  - `subsample=0.8` and `colsample_bytree=0.8` were used to improve generalization.
  - `tree_method='gpu_hist'` for GPU acceleration.
- **Training**: The model was trained with an 80:20 split between training and validation sets, and cross-validation was used to assess consistency in performance metrics.

### **7. Evaluation Metrics**

The model was evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: The model achieved an MAE of approximately 2.15e-05 on the validation set, indicating a very low average error in predicting crime hour.
- **Root Mean Squared Error (RMSE)**: The RMSE value was approximately 3.91e-05, indicating good overall model fit.
- **R-squared (R²)**: The model's R² score was almost 1, indicating that the model explains almost all the variance in the target variable. However, such a high R² score might suggest overfitting, and further testing on unseen data was necessary.
- **Cross-Validation**: Cross-validation with 3 folds yielded a Mean Absolute Error (MAE) of approximately 0.00006, showing consistent performance across different subsets of the data.

### **8. Additional Analysis and Interpretability**

- **Residual Analysis**: Residual plots were analyzed to confirm that errors were distributed randomly, with no visible patterns indicating model biases.
- **Feature Importance**: XGBoost's feature importance and SHAP values were used to understand the contributions of different features. Key influential features included geographical clusters, crime type, and time of day.
- **Hotspot and Crime Type Analysis**: The model highlighted areas of the city where certain types of crime occurred more frequently at specific times. This information was used to derive insights into crime hotspots and the predominant types of crimes in those areas.

### **9. Model Deployment and Next Steps**

- **Deployment Considerations**: The trained XGBoost model is suitable for deployment due to its balance between accuracy and computational efficiency. GPU acceleration can be leveraged for real-time predictions.
- **Testing with Real-Time Data**: The model should be further tested with new data entered by users, such as entering a location and receiving predicted crime times and types, to ensure robustness.
- **Future Work**: Expand the model to include a classification aspect for crime type prediction, using approaches like Random Forest Classifiers, LSTMs, or Transformer-based models for better temporal modeling.

### **Conclusion**

The crime time prediction model showed high predictive performance based on the validation metrics obtained. However, care must be taken to ensure the model generalizes well to new, unseen data. Interpretability techniques like SHAP analysis and feature importance were crucial for understanding the drivers behind the model's predictions, providing valuable insights into crime patterns in Chicago.
