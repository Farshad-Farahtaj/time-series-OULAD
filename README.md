# Time Series Analysis on OULAD Dataset

## Overview

This project demonstrates a comprehensive approach to time series forecasting using the Open University Learning Analytics Dataset (OULAD), incorporating both traditional statistical methods and modern machine learning techniques.

## Authors
- Seyyed Alireza Khoshsolat (D03000041)
- Farshad Farahtaj (D03000028)
- Zahra Jafarinejad (D03000083)

## University
University of Naples Federico II

## Course
Hardware and Software for Big Data (Module B)

## Instructor
Professoressa Flora Amato

## Date
July 2, 2024

## Key Takeaways

1. **Data Preprocessing is Crucial**:
   - Proper handling of anomalies and ensuring data stationarity are foundational steps for accurate forecasting.

2. **Model Selection**:
   - Different models have varying strengths. While SARIMA and ARIMAX effectively captured seasonal patterns, Prophet and CNN models excelled in handling complex trends and multiple seasonalities.

3. **Feature Engineering**:
   - The inclusion of lag features significantly improved the performance of ARIMAX and CNN models, highlighting the importance of feature engineering in time series forecasting.

4. **Model Evaluation**:
   - Consistent evaluation using MAE and MAPE provided clear insights into model performance, guiding the selection of the best model.

## Project Structure

```
├── data/                  # Dataset directory (download data files from Kaggle)
│   └── README.md          # Instructions for downloading dataset
│
├── notebooks/             # Jupyter notebooks
│   └── Harware_and_Software_Mod_B_Final_Project.ipynb
│
├── src/                   # Source code
│   └── Hardware_and_sofware_Mod_B_Final_Project_Sreamlit_version.py
│
├── models/                # Trained models
│   └── tuner0.json
│
├── docs/                  # Documentation
│   └── Harware_and_Software_Mod_B_Final_Project.pdf
│
├── requirements.txt       # Project dependencies
│
└── README.md              # Project description
```

## Installation and Setup

1. Clone this repository
   ```bash
   git clone https://github.com/Alirezakhoshsolat/time-series-OULAD.git
   cd time-series-OULAD
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Jupyter Notebook

To explore the detailed analysis and model development:

```bash
jupyter notebook notebooks/Harware_and_Software_Mod_B_Final_Project.ipynb
```

### Streamlit Application

To run the interactive Streamlit application:

```bash
streamlit run src/Hardware_and_sofware_Mod_B_Final_Project_Sreamlit_version.py
```

## Dataset

This project uses the [Open University Learning Analytics Dataset (OULAD)](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad) from Kaggle, which contains information about students' interactions with the Virtual Learning Environment (VLE).

### Data Source
- **Dataset Name**: Open University Learning Analytics Dataset (OULAD)
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)

### Dataset Files
The dataset includes:
- **Courses**: Information about the modules and their presentations
- **Assessments**: Information about the assessments in each module
- **VLE**: Information about the materials in the VLE
- **Student Information**: Demographic information about the students
- **Student Registration**: Information about when students registered for modules
- **Student Assessment**: Information about student performance in assessments
- **Student VLE**: Information about student interactions with the VLE

### Data Access Instructions
Due to GitHub's file size limitations (especially for the `studentVle.csv` which is over 400MB), the raw data files are not included in this repository. To reproduce our analysis:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
2. Extract the files and place them in the `/data` directory of this project
3. Ensure all CSV files are named according to the original dataset naming convention

## Methodology

1. **Data Preprocessing**: Cleaning and transforming data for analysis
2. **Exploratory Data Analysis**: Understanding patterns in student interactions
3. **Time Series Analysis**: Applying various forecasting techniques
   - SARIMA (Seasonal AutoRegressive Integrated Moving Average)
   - ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables)
   - Prophet (Facebook's forecasting tool)
   - CNN (Convolutional Neural Network)
4. **Model Evaluation**: Comparing model performance using MAE and MAPE metrics

## Results

The project successfully demonstrates how different time series forecasting models can be applied to predict student interactions with virtual learning environments, with the CNN model achieving the best overall performance.

## License

[MIT License](LICENSE)
