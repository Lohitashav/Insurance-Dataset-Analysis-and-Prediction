# Insurance-Dataset-Analysis-and-Prediction
This project explores and models an insurance dataset to analyze key factors affecting insurance charges and predict future values. The project is divided into two primary phases:

- Exploratory Data Analysis (EDA): A detailed analysis to uncover insights and relationships within the data.
- Model Training and Evaluation: Development and comparison of machine learning models to predict insurance charges.

# Project Structure
1. Exploratory Data Analysis (EDA)
   The EDA notebook performs data exploration and visualization to understand:
   - Key statistical metrics of the data.
   - Relationships between independent variables (e.g., age, BMI, region) and the target variable (insurance charges).
   - Insights derived from data distribution and feature importance.
     
2. Model Training and Evaluation
   The Model Training notebook focuses on:
   - Preprocessing the dataset (e.g., encoding categorical variables, scaling numerical features).
   - Implementing multiple regression models, including:
      - Linear Regression
      - Decision Trees
      - Random Forest
      - Gradient Boosting (CatBoost, XGBoost, AdaBoost)
      - Support Vector Machines (SVM)
   - Hyperparameter tuning using techniques like RandomizedSearchCV.
   - Evaluating model performance using metrics such as:
      - Mean Squared Error (MSE)
      - Mean Absolute Error (MAE)
      - R² Score
      - 
# Dataset
The dataset contains information about insurance charges and includes the following features:

- Age: Age of the individual.
- Sex: Gender of the individual.
- BMI: Body Mass Index.
- Children: Number of children.
- Smoker: Smoking status.
- Region: Residential region.
- Charges: Insurance charges (target variable).
  
# Tools and Libraries Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn (for visualization)
- Scikit-learn (for model development)
- XGBoost and CatBoost
- Jupyter Notebook
  
# How to Run
1. Clone the repository and navigate to the project directory.
2. Ensure the dataset (insurance.csv) is in the data/ folder.
3. Install the required libraries:
      pip install -r requirements.txt
4. Run the notebooks sequentially:
     -EDA.ipynb for data analysis.
     -model_training.ipynb for model development.
5. Key Insights and Results
- Insights derived from data analysis.
- Comparative evaluation of different regression models.
- Identification of the best-performing model for insurance charge prediction.

# Conclusion
This project showcases a complete machine learning pipeline, from data analysis to model evaluation, providing a hands-on experience with real-world data.
