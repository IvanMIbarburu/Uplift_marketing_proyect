# Uplift Marketing Optimization

This project demonstrates how uplift modeling can be used to optimize marketing targeting decisions by maximizing **incremental profit**, not just conversion rates.

## Problem
Traditional conversion models predict *who will convert*, but not *who should be targeted*.  
Uplift modeling estimates the **causal impact** of showing an ad.

## Data
This project uses the public Criteo Uplift Prediction Dataset from Kaggle [https://www.kaggle.com/code/lcw2099/criteo-uplift-modeling]
Due to size of the csv file and good practices, raw data is not included in the repository. Please download it and place it under 'data/raw' to reproduce the results.

## Approach
- Two-model uplift approach (treatment vs control)
- Logistic regression for ranking users (simplest model)
- Uplift and Qini evaluation (for sorting the users)
- Economic optimization using business parameters ("what-if" scenarios)


## Profit Optimization
We translate uplift into expected profit using:
- CPM 
- Impressions per user
- Revenue per conversion

The optimal targeting percentile is selected by maximizing incremental profit.

## Interactive App
A Streamlit app allows real-time "what-if"" analysis:
- Adjust business parameters
- Visualize profit curves
- Identify optimal targeting strategy

Link to app: (link to Streamlit Cloud)

## PDF presentation
Project presentation to show the results and the value of this analysis

## 🔮 Future Work
- Tree-based uplift models
- Multi-treatment scenarios
- Policy learning / bandits
