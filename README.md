# Dubai-Real-Estate-Price-Analysis
This repository contains code and report for the 'Dubai-Real-Estate-Price-Analysis' project.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Technical Overview](#technical-overview)
6. [Results](#results)
7. [Data Licensing](#data-licensing)

## Project Overview

This project explores residential property transactions in Dubai during 2025 and develops machine learning models to understand and predict property prices.

The analysis follows a complete data science workflow, including data preparation, exploratory data analysis (EDA), feature engineering, statistical testing, developer identification, and predictive modelling.

The primary modelling target is property price per square meter (AED/m²), with the objective of identifying the factors that most strongly influence residential property values across Dubai.

## Business Problem

Property prices are influenced by numerous factors, including location, property size, property type, developer reputation, accessibility, and local amenities.

The objective of this project is to:
 - Identify the main drivers of residential property prices.
 - Quantify the influence of structural and spatial characteristics.
 - Evaluate the impact of developers on property valuations.
 - Build predictive models capable of estimating property prices from transaction attributes.

## Dataset

The project uses the **DLD Transactions (Open)** dataset published by the Dubai Data & Statistics Establishment (DDSE).

The dataset contains residential property transactions recorded throughout 2025 and includes information such as:
 - Area
 - Property type
 - Property size
 - Number of bedrooms
 - Property price
 - Building / Project name
 - Transaction date
 - Nearby amenities

The raw dataset is not distributed through this repository and must be obtained directly from the official Dubai Open Data portal.

For dataset attribution and licensing information, see DATA_LICENSE.md.

## Methodology

The project is divided into five main stages:

1. **Data Preparation:**
   - Data cleaning and validation
   - Missing value handling
   - Feature creation
   - Log transformations
   - Outlier investigation

2. **Exploratory Data Analysis:**
   - Transaction distribution analysis
   - Area-level market analysis
   - Property type comparison
   - Bedroom configuration analysis
   - Temporal trends

3. **Feature Engineering:**
   - Developer extraction
   - Area-level market statistics
   - Amenitiy indicators
   - Log-transformed pricing variables

4. **Statistical Analysis:**
   - Mann-Whitney U tests
   - Kruskal-Wallis tests
   - Dunn post-hoc comparisons
   - Effect size calculations
   - Correlation analysis

5. **Predictive Modelliing:**
   - Baseline linear models
   - Regularised regression
   - Tree-based machine learning models
   - Model comparison and validation

## Technical Overview
### Technologies Used
- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Scikit-Learn
- LightGBM
- Matplotlib
- Jupyter Notebook
- Git & Github

### Skills Demonstrated
- Data Cleaning
- Exploratory Data Analysis
- Statistical Inference
- Feature Engineering
- Machine Learning
- Model Validation
- Data Visualisation
- Reproducible Research

## Results
Results and model performance metrics will be added as the modelling stage progresses.

## Data Licensing
This repository contains original code and documentation licensed under the MIT License.

The underlying dataset is owned by the Dubai Data & Statistics Establishment (DDSE) and is not included in this repository.

See DATA_LICENSE.md for details.
