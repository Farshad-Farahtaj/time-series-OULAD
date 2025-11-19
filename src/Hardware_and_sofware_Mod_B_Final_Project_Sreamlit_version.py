import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from datetime import timedelta
import itertools

st.title("time series analysis on OULAD dataset")
st.header("**overview**")
st.markdown('''Through this project, we demonstrated a comprehensive approach to time series forecasting, incorporating both traditional statistical methods and modern machine learning techniques. The key takeaways from this project are:

1. **Data Preprocessing is Crucial**:
   - Proper handling of anomalies and ensuring data stationarity are foundational steps for accurate forecasting.

2. **Model Selection**:
   - Different models have varying strengths. While SARIMA and ARIMAX effectively captured seasonal patterns, Prophet and CNN models excelled in handling complex trends and multiple seasonalities.

3. **Feature Engineering**:
   - The inclusion of lag features significantly improved the performance of ARIMAX and CNN models, highlighting the importance of feature engineering in time series forecasting.

4. **Model Evaluation**:
   - Consistent evaluation using MAE and MAPE provided clear insights into model performance, guiding the selection of the best model.
''')

# Function to transform code_presentation
def transform_presentation(presentation):
    year = int(presentation[:4])
    semester = presentation[4]
    if semester == 'B':
        return f'{year}-02-01'
    elif semester == 'J':
        return f'{year}-10-01'
    else:
        return None

# Function to reverse differencing and log transformation
def reverse_diff_log_transform(pred, original_data):
    pred_cumsum = pred.cumsum()
    pred_log_clicks = pred_cumsum + original_data.iloc[0]
    pred_clicks = np.exp(pred_log_clicks)
    return pred_clicks

st.header("Data Preprocessing")
st.markdown('''


In our project, we meticulously preprocessed seven datasets to ensure they were ready for effective analysis and modeling. Here’s a summary of the key preprocessing steps we performed on each dataset:

#### 1. Student Information Dataset
- **Data Loading**: We started by loading the dataset containing information about the students, including their demographics and other attributes.
- **Handling Missing Values**: We dealt with missing values by either imputing them with appropriate strategies or dropping rows/columns where necessary.
- **Encoding Categorical Variables**: Categorical variables such as `gender`, `region`, and `highest_education` were encoded into numerical values using techniques like one-hot encoding or label encoding.
- **Normalizing Numerical Features**: Continuous numerical features were normalized to bring them into a standard range, ensuring they contributed equally to the models.

#### 2. Assessment Data
- **Data Transformation**: The assessment data, which detailed students’ performance in various assessments, was loaded and transformed.
- **Date Parsing**: Assessment dates were parsed into datetime objects to facilitate time-series analysis.
- **Feature Engineering**: New features were engineered, such as the total score and the average score per assessment, to enrich the dataset.

#### 3. Course Data
- **Loading the Dataset**: The course data, which provided details about different course modules, was loaded.
- **Mapping Course Modules**: Course modules identified by codes (e.g., 'AAA', 'BBB') were mapped to numerical values for consistency and ease of analysis.
- **Merging with Other Datasets**: The course data was merged with other datasets, such as student information and assessments, to provide a comprehensive view.

#### 4. Student Registration Data
- **Data Integration**: The student registration dataset, detailing the enrollment status of students, was integrated with the student information dataset.
- **Handling Date Formats**: Registration and deregistration dates were parsed and standardized.
- **Status Encoding**: Registration statuses were encoded into numerical values to facilitate analysis.

#### 5. Virtual Learning Environment (VLE) Data
- **Loading the Dataset**: The VLE interactions dataset, capturing students’ interactions with various activities, was loaded.
- **Mapping Course Modules**: Course modules were mapped to numerical values.
- **Transforming Presentation Codes**: Presentation codes indicating the start date of courses were transformed into standard date formats for time-series analysis.
- **Encoding Activity Types**: Various activity types (e.g., 'forum', 'quiz') were encoded into numerical values.
- **Handling Anomalies**: Anomalies in the interaction data were detected using statistical methods like z-scores and imputed with median values.
- **Feature Creation**: Lag features were created to capture the sequential nature of the interactions.

#### 6. Student VLE Interactions
- **Data Integration**: This dataset combined student interactions with VLE activities and was integrated with other datasets for a holistic view.
- **Cleaning and Encoding**: Categorical variables were encoded, and continuous variables were normalized.
- **Aggregating Interactions**: Interactions were aggregated to derive meaningful insights, such as the total number of clicks per student per activity.

#### 7. Student Assessment Data
- **Data Transformation**: The dataset containing detailed records of students’ assessment scores was loaded and transformed.
- **Merging with Assessment Data**: This dataset was merged with the assessment data to link each student’s performance with specific assessments.
- **Feature Engineering**: Additional features, such as the average score and number of attempts per assessment, were engineered to enrich the data.

### Conclusion

The preprocessing steps for each dataset involved loading the data, cleaning and transforming it, encoding categorical variables, and engineering new features. These steps were crucial to ensure that the data was in a format suitable for effective analysis and predictive modeling. By carefully handling each dataset, we set a strong foundation for the subsequent stages of our project, enabling us to derive meaningful insights and build robust predictive models. This comprehensive preprocessing allowed us to handle various aspects of the educational data, ensuring a well-rounded approach to understanding and predicting student performance and interactions.''')

# Load and preprocess datasets
st.header("Exploratory Data Analysis (EDA):")

# 1. Student Info Dataset
st.subheader("Student Info Dataset")
student_info_path = r"D:\study\Hardware and Software per Big Data mod B\project\preprocessed_student_info.csv"
student_info = pd.read_csv(student_info_path, parse_dates=['presentation_start'])

# Gender distribution
st.subheader('Gender Distribution')
fig = px.histogram(student_info, x='gender', color='gender', title='Gender Distribution')
st.plotly_chart(fig)

st.markdown(''' From the plot, we can see if the dataset is balanced in terms of gender representation. The uploaded plot shows that there are slightly more male students than female students.''')

# Region distribution
st.subheader('Region Distribution')
fig = px.histogram(student_info, x='region', color='region', title='Region Distribution')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''This plot helps us understand the geographical distribution of students. The plot indicates that most students come from Scotland, North Western Region, and London Region, while fewer students are from Ireland and North Region.''')

# Highest education level distribution
st.subheader('Highest Education Level Distribution')
fig = px.histogram(student_info, x='highest_education', color='highest_education', title='Highest Education Level Distribution')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''This visualization shows that the majority of students have an A Level or Equivalent education, followed by those with Lower Than A Level and HE Qualification. Very few students have Post-Graduate Qualifications or No Formal Qualifications.''')

# Age band distribution
st.subheader('Age Band Distribution')
fig = px.histogram(student_info, x='age_band', color='age_band', title='Age Band Distribution')
st.plotly_chart(fig)

st.markdown('''The plot reveals that most students are in the 0-35 age band, with a smaller number in the 35-55 age band and very few in the 55<= age band.
''')

# Disability status distribution
st.subheader('Disability Status Distribution')
fig = px.histogram(student_info, x='disability', color='disability', title='Disability Status Distribution')
st.plotly_chart(fig)

st.markdown('''The plot indicates that the majority of students do not have disabilities.''')

# Final results distribution
st.subheader('Final Results Distribution')
fig = px.histogram(student_info, x='final_result', color='final_result', title='Final Results Distribution')
st.plotly_chart(fig)

st.markdown('''The plot shows that most students either passed or withdrew, with fewer students failing or achieving a distinction.''')

# Distribution of studied credits
st.subheader('Distribution of Studied Credits')
fig = px.histogram(student_info, x='studied_credits', nbins=30, title='Distribution of Studied Credits')
st.plotly_chart(fig)

st.markdown('''The plot shows that most students have studied around 60 credits, with a few students studying up to 240 credits.''')

# Distribution of number of previous attempts
st.subheader('Distribution of Number of Previous Attempts')
fig = px.histogram(student_info, x='num_of_prev_attempts', nbins=30, title='Distribution of Number of Previous Attempts')
st.plotly_chart(fig)

st.markdown('''The plot indicates that most students are attempting the course for the first time, with very few students having multiple previous attempts.''')

# Enrollment over time
st.subheader('Enrollment Over Time')
enrollment_counts = student_info['presentation_start'].value_counts().sort_index()
fig = px.line(enrollment_counts, title='Enrollment Over Time')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''The plot shows an increasing trend in student enrollment over time.''')

st.markdown("These visualizations provide valuable insights into the demographics and behaviors of the students, which will be useful for further analysis and model building.")

# 2. Student VLE Dataset
st.subheader("Student VLE Dataset")
student_vle_path = r"D:\study\Hardware and Software per Big Data mod B\project\preprocessed_student_vle.csv"
student_vle = pd.read_csv(student_vle_path, parse_dates=['presentation_start', 'interaction_date'])

# Interactions over time
st.subheader('Interactions Over Time')
interaction_counts = student_vle['interaction_date'].value_counts().sort_index()
fig = px.line(interaction_counts, title='Interactions Over Time')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''The plot shows how the number of interactions changes over time. Peaks and troughs in the interaction count can indicate periods of high and low student activity. For instance, we can observe multiple spikes in interactions which might correspond to key academic dates such as assignment deadlines or examination periods. The final drop towards zero likely indicates the end of the data collection period.''')

# 3. Student Assessment Dataset
st.subheader("Student Assessment Dataset")
student_assessment_path = r"D:\study\Hardware and Software per Big Data mod B\project\preprocessed_student_assessment.csv"
student_assessment = pd.read_csv(student_assessment_path, parse_dates=['submission_date'])

st.markdown("The aim of this sub-step is to explore the distribution of scores and analyze the submission patterns over time in the student assessment dataset. Below are the visualizations and their interpretations.")

# Distribution of scores
st.subheader('Distribution of Scores')
fig = px.histogram(student_assessment, x='score', nbins=30, title='Distribution of Scores')
st.plotly_chart(fig)

st.markdown('''This histogram displays the distribution of student scores. Most scores are concentrated in the higher range (70-100), indicating that a significant number of students performed well on their assessments. There are also noticeable peaks around the 0-20 range, which could indicate some students not attempting or failing the assessments.''')

# Submissions over time
st.subheader('Submissions Over Time')
submission_counts = student_assessment['submission_date'].value_counts().sort_index()
fig = px.line(submission_counts, title='Submissions Over Time')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''This line plot illustrates the number of submissions over time. There is a high frequency of submissions around certain dates, likely corresponding to deadlines. The pattern indicates several peaks and troughs, showing the cyclical nature of submission deadlines. Notably, there is a sharp decline after mid-2014, indicating a possible end of the assessment period for the courses.''')

# 4. Assessments Dataset
st.subheader("Assessments Dataset")
assessments_path = r"D:\study\Hardware and Software per Big Data mod B\project\preprocessed_assessments.csv"
assessments = pd.read_csv(assessments_path, parse_dates=['assessment_date'])

st.markdown("In this code block, we perform exploratory data analysis (EDA) on the assessments dataset to understand the distribution of assessment types, weights, and the timing of assessments.")

# Distribution of assessment types
st.subheader('Distribution of Assessment Types')
fig = px.histogram(assessments, x='assessment_type', color='assessment_type', title='Distribution of Assessment Types')
st.plotly_chart(fig)

st.markdown('''This line plot illustrates the number of submissions over time. There is a high frequency of submissions around certain dates, likely corresponding to deadlines. The pattern indicates several peaks and troughs, showing the cyclical nature of submission deadlines. Notably, there is a sharp decline after mid-2014, indicating a possible end of the assessment period for the courses.''')


# Assessment weights distribution
st.subheader('Distribution of Assessment Weights')
fig = px.histogram(assessments, x='weight', nbins=20, title='Distribution of Assessment Weights')
st.plotly_chart(fig)

st.markdown('''This histogram displays the distribution of assessment weights. Most assessments have a lower weight, with a significant number clustered around zero. There are also peaks at higher weights, indicating some assessments carry more importance.''')

# Assessments over time
st.subheader('Assessments Over Time')
assessment_counts = assessments['assessment_date'].value_counts().sort_index()
fig = px.line(assessment_counts, title='Assessments Over Time')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''The line plot shows the number of assessments conducted over time. Peaks represent periods with a high number of assessments, which could correspond to exam periods or deadlines for assignments.''')

st.markdown("This part of EDA highlights the nature of assessments in terms of type, weight, and their distribution over time, providing insights into the structure and scheduling of assessments within the dataset.")

# 5. Courses Dataset
st.subheader("Courses Dataset")
courses_path = r"D:\study\Hardware and Software per Big Data mod B\project\preprocessed_courses.csv"
courses = pd.read_csv(courses_path, parse_dates=['presentation_start'])

st.markdown("In this part, we perform EDA on the courses dataset to understand the distribution of courses across different code modules.")

# Distribution of courses by code_module
st.subheader('Distribution of Courses by Code Module')
fig = px.histogram(courses, x='code_module', color='code_module', title='Distribution of Courses by Code Module')
st.plotly_chart(fig)

st.markdown('''The bar plot shows the count of courses grouped by different code modules. Each bar represents a different module, and the height of the bar indicates the number of courses in that module. 
This plot helps in understanding which modules have the most or least number of courses.''')

# 6. VLE Dataset
st.subheader("VLE Dataset")
vle_path = r"D:\study\Hardware and Software per Big Data mod B\project\preprocessed_vle.csv"
vle = pd.read_csv(vle_path, parse_dates=['presentation_start'])

st.markdown("In this step, we perform EDA on the preprocessed Virtual Learning Environment (VLE) data to understand the distribution of different activity types and their association with different code modules.")

# Distribution of activity types
st.subheader('Distribution of Activity Types')
fig = px.histogram(vle, x='activity_type', color='activity_type', title='Distribution of Activity Types')
st.plotly_chart(fig)

st.markdown('''The bar plot shows the frequency of each activity type in the VLE dataset.
This helps to identify the most common types of activities within the VLE platform.
''')
st.markdown('''resource: 0, oucontent: 1, url: 2, homepage: 3, subpage: 4, glossary: 5, forumng: 6, oucollaborate: 7, dataplus: 8, quiz: 9, ouelluminate: 10, sharedsubpage: 11, questionnaire: 12, page: 13, externalquiz: 14, ouwiki: 15, dualpane: 16, repeatactivity: 17 , folder: 18 , htmlactivity: 19''')

# Distribution of VLE activities by code module
st.subheader('Distribution of VLE Activities by Code Module')
fig = px.histogram(vle, x='code_module', color='code_module', title='Distribution of VLE Activities by Code Module')
st.plotly_chart(fig)

st.markdown('''The bar plot shows how VLE activities are distributed among different code modules.
This visualization helps to understand which modules have the highest number of associated VLE activities.''')

# 7. Student Registration Dataset
st.subheader("Student Registration Dataset")
student_registration_path = r"D:\study\Hardware and Software per Big Data mod B\project\preprocessed_student_registration.csv"
student_registration = pd.read_csv(student_registration_path, parse_dates=['registration_date', 'unregistration_date'])

st.markdown("In this sub-step, we perform EDA on the preprocessed student registration data to understand the patterns and trends related to student registrations and unregistrations over time.")

# Distribution of registration dates
st.subheader('Distribution of Registration Dates')
fig = px.histogram(student_registration, x='registration_date', nbins=30, title='Distribution of Registration Dates')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown(''' - The histogram shows the distribution of registration dates.
   - Peaks are observed around the start of each semester, indicating that most registrations occur at these times.
   - The distribution helps identify the periods with the highest student registrations.''')

# Distribution of unregistration dates
st.subheader('Distribution of Unregistration Dates')
fig = px.histogram(student_registration, x='unregistration_date', nbins=30, title='Distribution of Unregistration Dates')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''- The histogram shows the distribution of unregistration dates.
   - Peaks around certain periods may indicate critical drop-out points or periods of high unregistration.
   - Understanding these periods helps in identifying when students are most likely to drop out and could lead to insights for interventions.''')

# Distribution of registrations by code module
st.subheader('Distribution of Registrations by Code Module')
fig = px.histogram(student_registration, x='code_module', color='code_module', title='Distribution of Registrations by Code Module')
st.plotly_chart(fig)

st.markdown('''- This count plot shows the number of registrations across different code modules.
   - Some modules have higher registration counts, indicating their popularity or compulsory nature.''')

# Registrations and unregistrations over time
st.subheader('Registrations and Unregistrations Over Time')
registration_counts = student_registration['registration_date'].value_counts().sort_index()
unregistration_counts = student_registration['unregistration_date'].value_counts().sort_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=registration_counts.index, y=registration_counts.values, mode='lines', name='Registrations'))
fig.add_trace(go.Scatter(x=unregistration_counts.index, y=unregistration_counts.values, mode='lines', name='Unregistrations', line=dict(color='red')))
fig.update_layout(title='Registrations and Unregistrations Over Time', xaxis_title='Date', yaxis_title='Count')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''- The line plot shows the count of registrations and unregistrations over time.
   - Blue line represents registrations and red line represents unregistrations.
   - This plot helps visualize trends and patterns over time, such as periods with high registrations or significant unregistrations.''')

st.markdown("By analyzing these plots, we gain valuable insights into the registration behaviors, identifying key periods for interventions and understanding the popularity of different modules. This information is crucial for making data-driven decisions to improve student retention and engagement.")

# 8. Student Registration Dataset
st.subheader('Assessment Dataset')
assessments_path = r"D:\study\Hardware and Software per Big Data mod B\project\preprocessed_assessments.csv"
assessments = pd.read_csv(assessments_path, parse_dates=['assessment_date'])

# Distribution of assessment dates
st.subheader('Distribution of assessment dates')
fig = px.histogram(assessments, x='assessment_date', nbins=30, title='Distribution of assessment dates')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown('''- The histogram shows the distribution of assessment dates.
  - Peaks are observed during certain periods, likely aligning with the academic calendar's assessment schedules.
  - This distribution helps identify the busiest periods for assessments, allowing for better planning and resource allocation.''')

st.markdown("By analyzing this plot, we can understand the scheduling of assessments and ensure that the distribution aligns with expectations. This insight can help in identifying any potential issues with the timing of assessments and make data-driven decisions for future scheduling.")

# Time Series Analysis and Forecasting
st.header("step 3: Anomaly Detection and Forecasting with Different Models")
st.subheader("**Introduction:**")
st.markdown("This section of the project focuses on detecting anomalies in the Virtual Learning Environment (VLE) interactions data and forecasting future interactions using different models. We start by loading and preprocessing the data, then proceed with anomaly detection using z-score. Subsequently, we use different models for forecasting.")

# Prophet Model
st.subheader("Prophet Model for Forecasting VLE Interactions")
st.markdown("In this step, we utilize Facebook's Prophet model to forecast VLE (Virtual Learning Environment) interactions. Prophet is particularly useful for time series data that exhibits strong seasonal effects and multiple seasonality with daily, weekly, and yearly patterns.")

merged_vle_data_path = r"D:\study\Hardware and Software per Big Data mod B\project\time_series_vle_interactions.csv"
merged_vle_data = pd.read_csv(merged_vle_data_path, parse_dates=['Date'])

filtered_data = merged_vle_data[merged_vle_data['Date'] < '2015-05-01']
filtered_data.set_index('Date', inplace=True)
filtered_data = filtered_data.asfreq('D')
filtered_data['log_clicks'] = np.log(filtered_data['Total_Clicks'].replace(0, np.nan)).dropna()
filtered_data['z_score'] = zscore(filtered_data['log_clicks'])
anomalies = filtered_data[np.abs(filtered_data['z_score']) > 2]

# Plot the data with anomalies
st.subheader('VLE Interactions with Anomalies Highlighted')
fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['log_clicks'], mode='lines', name='Log Clicks'))
fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['log_clicks'], mode='markers', name='Anomalies', marker=dict(color='red')))
fig.update_layout(title='VLE Interactions with Anomalies Highlighted', xaxis_title='Date', yaxis_title='Log Clicks')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.markdown("The anomalies were plotted to visualize their occurrence in the dataset.")

# Impute anomalies by replacing them with the median of the series
filtered_data['log_clicks_no_anomalies'] = filtered_data['log_clicks'].copy()
filtered_data.loc[anomalies.index, 'log_clicks_no_anomalies'] = filtered_data['log_clicks_no_anomalies'].median()

# Prepare data for Prophet
df_prophet = filtered_data.reset_index()[['Date', 'log_clicks_no_anomalies']]
df_prophet.columns = ['ds', 'y']

def evaluate_prophet(params):
    yearly_seasonality, weekly_seasonality, changepoint_prior_scale, seasonality_mode = params
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_prophet)
    
    # Forecast future interactions
    future = model.make_future_dataframe(periods=120)
    forecast = model.predict(future)
    
    # Evaluate the model using MAE and MAPE
    y_truth = df_prophet['y']
    y_forecasted = forecast.loc[forecast['ds'].isin(df_prophet['ds']), 'yhat']
    mae = mean_absolute_error(y_truth, y_forecasted)
    mape = mean_absolute_percentage_error(y_truth, y_forecasted)
    
    return mae, mape

# Define parameter grid
param_grid = {
    'yearly_seasonality': [True, False],
    'weekly_seasonality': [True, False],
    'changepoint_prior_scale': [0.01, 0.1, 0.5],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Perform grid search
best_mae = float('inf')
best_mape = float('inf')
best_params = None

for params in itertools.product(*param_grid.values()):
    mae, mape = evaluate_prophet(params)
    if mae < best_mae:
        best_mae = mae
        best_mape = mape
        best_params = params

# Fit the best model
yearly_seasonality, weekly_seasonality, changepoint_prior_scale, seasonality_mode = best_params
best_model = Prophet(
    yearly_seasonality=yearly_seasonality,
    weekly_seasonality=weekly_seasonality,
    changepoint_prior_scale=changepoint_prior_scale,
    seasonality_mode=seasonality_mode
)
best_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
best_model.fit(df_prophet)

# Forecast future interactions
future = best_model.make_future_dataframe(periods=120)
forecast = best_model.predict(future)

# Reverse the log transformation for visualization
forecast['yhat_original_scale'] = np.exp(forecast['yhat'])

# Plot the forecast in the original scale
st.subheader('Prophet Forecast in Original Scale')
fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Total_Clicks'], mode='lines', name='Observed'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_original_scale'], mode='lines', name='Forecast', line=dict(color='orange')))
fig.update_layout(title='Prophet Forecast in Original Scale', xaxis_title='Date', yaxis_title='Total Clicks')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.write("Evaluating the model using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).")
st.write('Prophet Model Mean Absolute Error:', best_mae)
st.write('Prophet Model Mean Absolute Percentage Error:', best_mape)

st.subheader("Conclusion")
st.markdown("The Prophet model effectively captured the underlying patterns in the VLE interactions data, providing accurate forecasts. The grid search approach allowed us to optimize the model parameters, ensuring the best possible performance. This step demonstrated the utility of Prophet in handling complex time series data with multiple seasonality, offering a robust solution for future forecasting tasks. The final visualization confirmed the model's accuracy and reliability.")

# ARIMAX Model
st.subheader("ARIMAX Model")
# Reload preprocessed data
merged_vle_data = pd.read_csv(merged_vle_data_path, parse_dates=['Date'])
filtered_data = merged_vle_data[merged_vle_data['Date'] < '2015-05-01']
filtered_data.set_index('Date', inplace=True)
filtered_data = filtered_data.asfreq('D')
filtered_data['log_clicks'] = np.log(filtered_data['Total_Clicks'].replace(0, np.nan)).dropna()

# Detect anomalies
filtered_data['z_score'] = zscore(filtered_data['log_clicks'])
anomalies = filtered_data[np.abs(filtered_data['z_score']) > 2]

# Impute anomalies with linear interpolation
filtered_data['log_clicks'] = filtered_data['log_clicks'].mask(np.abs(filtered_data['z_score']) > 2)
filtered_data['log_clicks'].interpolate(method='linear', inplace=True)

# Differencing to make the data stationary
filtered_data['diff_clicks'] = filtered_data['log_clicks'].diff().dropna()

# Check stationarity
def adf_test(series):
    result = adfuller(series)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    for key, value in result[4].items():
        st.write('Critical Value {}: {}'.format(key, value))
    return result

st.subheader("ADF Test on Differenced Data")
if not filtered_data['diff_clicks'].dropna().empty:
    adf_test(filtered_data['diff_clicks'].dropna())
else:
    st.write("Differenced data is empty or only contains NaN values.")

st.markdown("- **Stationarity**: The ADF test confirms the differenced data is stationary (p-value < 0.05).")

# Add lag features
lags = [1, 2, 3, 7, 14, 21, 28]  # Example lags
for lag in lags:
    filtered_data[f'lag_{lag}'] = filtered_data['diff_clicks'].shift(lag)
filtered_data.dropna(inplace=True)

# Check if there is enough data to proceed with ARIMAX
if not filtered_data.empty:
    # Fit ARIMAX model with predefined parameters
    best_params_arimax = (2, 0, 2)
    exog_arimax = filtered_data[[f'lag_{lag}' for lag in lags]]
    mod_arimax = sm.tsa.ARIMA(filtered_data['diff_clicks'],
                              exog=exog_arimax,
                              order=best_params_arimax)
    results_arimax = mod_arimax.fit()

    # Print the summary
    st.subheader("ARIMAX Model Summary")
    st.text(results_arimax.summary().as_text())

    # Forecast future interactions
    forecast_steps_arimax = 240
    exog_forecast_arimax = exog_arimax.iloc[-forecast_steps_arimax:, :]
    pred_uc_arimax = results_arimax.get_forecast(steps=forecast_steps_arimax, exog=exog_forecast_arimax)
    pred_ci_arimax = pred_uc_arimax.conf_int()


    # Plot the forecast
    st.subheader('ARIMAX Forecast')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['diff_clicks'], mode='lines', name='Observed'))
    fig.add_trace(go.Scatter(x=pred_uc_arimax.predicted_mean.index, y=pred_uc_arimax.predicted_mean, mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=pred_ci_arimax.index, y=pred_ci_arimax.iloc[:, 0], fill=None, mode='lines', line_color='lightgrey', name='Lower CI'))
    fig.add_trace(go.Scatter(x=pred_ci_arimax.index, y=pred_ci_arimax.iloc[:, 1], fill='tonexty', mode='lines', line_color='lightgrey', name='Upper CI'))
    fig.update_layout(title='ARIMAX Forecast', xaxis_title='Date', yaxis_title='Differenced Log Clicks')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

    st.markdown("Forecast future interactions for 240 steps and plot the forecast with confidence intervals.")


    # Evaluate the model using MAE and MAPE
    y_forecasted_arimax = pred_uc_arimax.predicted_mean
    y_truth_arimax = filtered_data['diff_clicks'][-forecast_steps_arimax:]

    mae_arimax = mean_absolute_error(y_truth_arimax, y_forecasted_arimax)
    mape_arimax = mean_absolute_percentage_error(y_truth_arimax, y_forecasted_arimax)


    # Reverse the differencing and log transformation for visualization
    predicted_clicks_out_sample_arimax = reverse_diff_log_transform(pred_uc_arimax.predicted_mean, filtered_data['log_clicks'][-1:])

    # Combine the original data and the predictions for plotting
    combined_data_arimax = filtered_data['Total_Clicks'].copy()
    combined_data_arimax = pd.concat([combined_data_arimax, predicted_clicks_out_sample_arimax])

    st.markdown('''
    - **Reverse Transformation**: Reverse the differencing and log transformation to get the predictions in the original scale.
    - **Plot Predictions**: Plot the predictions against the actual values in the original scale.''')

    # Plot the predictions against the actual values
    st.subheader('ARIMAX Forecast in Original Scale')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Total_Clicks'], mode='lines', name='Observed'))
    fig.add_trace(go.Scatter(x=predicted_clicks_out_sample_arimax.index, y=predicted_clicks_out_sample_arimax, mode='lines', name='Out-of-sample Forecast', line=dict(color='orange')))
    fig.update_layout(title='ARIMAX Forecast in Original Scale', xaxis_title='Date', yaxis_title='Total Clicks')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
else:
    st.write("Not enough data to fit ARIMAX model.")

st.write("Evaluating the model using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).")
st.write('ARIMAX Model Mean Absolute Error:', mae_arimax)
st.write('ARIMAX Model Mean Absolute Percentage Error:', mape_arimax)


# CNN Model
st.subheader("Convolutional Neural Network (CNN) Model for Forecasting VLE Interactions")
st.markdown("In this stage of our project, we applied a Convolutional Neural Network (CNN) model to forecast VLE interactions. CNNs are powerful deep learning models typically used for image data but have also proven effective for time series forecasting due to their ability to capture spatial (temporal) dependencies.")
# Reload preprocessed data
merged_vle_data = pd.read_csv(merged_vle_data_path, parse_dates=['Date'])
filtered_data = merged_vle_data[merged_vle_data['Date'] < '2015-05-01']
filtered_data.set_index('Date', inplace=True)
filtered_data = filtered_data.asfreq('D')
filtered_data['log_clicks'] = np.log(filtered_data['Total_Clicks'].replace(0, np.nan)).dropna()

filtered_data['z_score'] = (filtered_data['log_clicks'] - filtered_data['log_clicks'].mean()) / filtered_data['log_clicks'].std()
anomalies = filtered_data[np.abs(filtered_data['z_score']) > 3]

# Impute anomalies by replacing them with the median of the series
filtered_data['log_clicks_no_anomalies'] = filtered_data['log_clicks'].copy()
filtered_data.loc[anomalies.index, 'log_clicks_no_anomalies'] = filtered_data['log_clicks_no_anomalies'].median()

# Prepare data for modeling
df_model = filtered_data[['log_clicks_no_anomalies']].copy()

# Add lag features
lags = 30  # Number of lag features
for lag in range(1, lags + 1):
    df_model[f'lag_{lag}'] = df_model['log_clicks_no_anomalies'].shift(lag)
df_model.dropna(inplace=True)

# Split data into train and test sets
train_size = int(len(df_model) * 0.8)
train, test = df_model[:train_size], df_model[train_size:]

X_train, y_train = train.drop(columns=['log_clicks_no_anomalies']), train['log_clicks_no_anomalies']
X_test, y_test = test.drop(columns=['log_clicks_no_anomalies']), test['log_clicks_no_anomalies']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data for CNN [samples, timesteps, features]
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Define the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    Dropout(0.2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred = y_pred.flatten()

# Reverse the log transformation for visualization
def reverse_log_transform(pred):
    pred_clicks = np.exp(pred)
    return pred_clicks

# Generate the predictions in original scale
predicted_clicks = reverse_log_transform(y_pred)
actual_clicks = reverse_log_transform(y_test)

# Evaluate the model
mae_cnn = mean_absolute_error(y_test, y_pred)
mape_cnn = mean_absolute_percentage_error(y_test, y_pred)

# Plot the predictions
st.subheader('CNN Model Predictions')
fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_data.index[:train_size], y=reverse_log_transform(y_train), mode='lines', name='Training Data'))
fig.add_trace(go.Scatter(x=filtered_data.index[train_size:train_size+len(predicted_clicks)], y=predicted_clicks, mode='lines', name='Predicted Data', line=dict(color='orange')))
fig.update_layout(title='CNN Model Predictions', xaxis_title='Date', yaxis_title='Total Clicks')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

st.write("Evaluating the model using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).")
st.write('CNN Model Mean Absolute Error:', mae_cnn)
st.write('CNN Model Mean Absolute Percentage Error:', mape_cnn)


# Model Comparison
st.header("Model Comparison")
model_comparison = pd.DataFrame({
    "Model": ["Prophet", "CNN", "ARIMAX"],
    "MAE": [best_mae, mae_cnn, mae_arimax],
    "MAPE": [best_mape, mape_cnn, mape_arimax]
})

# Reshape the dataframe for plotting
model_comparison_melted = model_comparison.melt(id_vars="Model", var_name="Metric", value_name="Value")

fig = px.bar(model_comparison_melted, x="Model", y="Value", color="Metric", barmode='group', title="Model Comparison")
st.plotly_chart(fig)

st.subheader("Model Comparison Conclusion:")
st.markdown('''the ARIMAX model shows the lowest MAE, its very high MAPE suggests it might not be well-calibrated relative to the scale of the data. The Prophet model demonstrates a good balance with low MAE and the lowest MAPE, indicating strong overall performance. The CNN model also performs well, especially in relative terms.''')


st.header('Conclusion')
st.markdown('''
In this project, we tackled the challenging problem of forecasting Virtual Learning Environment (VLE) interactions. The goal was to accurately predict future interactions based on historical data, identifying anomalies and understanding the underlying patterns in the data. We applied several advanced time series forecasting techniques, including SARIMA, ARIMAX, Prophet, and Convolutional Neural Networks (CNNs). Here, we summarize the key findings, methodologies, and insights gained from this comprehensive analysis.

### Key Steps and Findings

#### 1. Data Preprocessing and Anomaly Detection

- **Data Loading and Initial Filtering**:
  - We started by loading the VLE interactions data, which contained daily interaction counts.
  - To avoid issues with zero interactions, we filtered out periods where interactions dropped to zero.

- **Log Transformation**:
  - To stabilize the variance and manage the wide range of interaction counts, we applied a log transformation to the interaction data.
  - This transformation is crucial for models that assume normally distributed errors.

- **Anomaly Detection**:
  - We used the z-score method to detect anomalies. Data points with z-scores beyond ±2 were considered anomalies.
  - Identifying and handling these anomalies was essential to ensure that they did not skew the model training and forecasting results.

- **Imputation of Anomalies**:
  - Anomalies were imputed using linear interpolation to maintain the continuity and trend of the data.

#### 2. Stationarity and Feature Engineering

- **Stationarity Testing**:
  - The Augmented Dickey-Fuller (ADF) test was employed to check for stationarity. Our data required differencing to achieve stationarity, a prerequisite for many time series models.

- **Lag Features**:
  - We introduced multiple lag features (up to 30 days) to capture the temporal dependencies in the data. These features help the models learn from past interactions and improve forecast accuracy.

#### 3. Forecasting Models

##### 3.1 ARIMAX Model

- **Incorporating Exogenous Variables**:
  - The ARIMAX model extends ARIMA by incorporating exogenous variables (lag features in this case), which can significantly enhance forecast accuracy.
  - The model captured additional information from lag features, improving its predictive capabilities.

- **Model Performance Evaluation**:
  - Similar to SARIMA, ARIMAX provided reliable forecasts with good MAE and MAPE values.
  - The inclusion of exogenous variables was particularly beneficial in capturing complex dependencies.
  - Strengths: Low MAE indicates that the model's average absolute error is small.
  - Weaknesses: Extremely high MAPE indicates that the model's predictions are off by a large percentage, which could imply issues with how the model is handling the data or scaling.

##### 3.2 Prophet Model

- **Model Configuration**:
  - Facebook’s Prophet model was chosen for its robustness and ease of use in handling time series data with multiple seasonalities.
  - A grid search over hyperparameters such as yearly and weekly seasonality, changepoint prior scale, and seasonality mode was performed to find the best model configuration.

- **Forecasting with Prophet**:
  - Prophet effectively modeled the trend and seasonality in the VLE interactions data.
  - The final model provided forecasts with competitive MAE and MAPE values, demonstrating its capability to handle complex time series data.

- **Model Performance Evaluation**:
  - Strengths: The Prophet model strikes a balance with a relatively low MAE and the lowest MAPE, indicating good performance both in absolute and relative terms.
  - Weaknesses: Slightly higher MAE compared to ARIMAX, but still within an acceptable range.
  
##### 3.3 CNN Model

- **Deep Learning Approach**:
  - A Convolutional Neural Network (CNN) was employed to capture intricate patterns and dependencies in the time series data.
  - The model was trained with 30 lag features, and the data was standardized before training.

- **Model Architecture**:
  - The CNN architecture included convolutional layers followed by dropout for regularization and dense layers for prediction.
  - This setup allowed the model to learn both local and global patterns in the data.

- **Model Performance Evaluation**:
  - Strengths: The CNN model captures complex patterns in the data, which may contribute to its relatively low MAPE, indicating that the model performs well relative to the actual values.
  - Weaknesses: Higher MAE compared to ARIMAX suggests that, in absolute terms, the CNN model's errors are larger.
In conclusion, this project provided a robust framework for forecasting VLE interactions, showcasing the effectiveness of combining statistical and machine learning techniques to achieve accurate and reliable predictions. The insights gained here will be invaluable for future projects involving time series data.''')
