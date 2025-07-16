import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def u_train_rich_poor_classifier():
    """
    LOADS THE PURCHASE DATA, LABELS CUSTOMERS AS RICH OR POOR,
    TRAINS A CLASSIFIER, AND EVALUATES ITS ACCURACY
    """

    # LOAD DATA FROM EXCEL FILE i.e. CUSTOMER PURCHASE BEHAVIOR
    u_excel_data = pd.read_excel(
        r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx", 
        sheet_name="Purchase data"
    )

    # CREATE LABEL COLUMN BASED ON PAYMENT i.e. RICH > 200, POOR ≤ 200
    u_excel_data['u_label_class'] = u_excel_data['Payment (Rs)'].apply(lambda x: 1 if x > 200 else 0)

    # SELECT FEATURES i.e. PRODUCT QUANTITIES
    u_features_X = u_excel_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()

    # SELECT TARGET i.e. RICH/POOR LABEL
    u_target_y = u_excel_data['u_label_class'].to_numpy()

    # SPLIT DATA FOR TRAINING AND TESTING i.e. 70% TRAIN, 30% TEST
    u_X_train, u_X_test, u_y_train, u_y_test = train_test_split(
        u_features_X, u_target_y, test_size=0.3, random_state=1
    )

    # TRAIN LOGISTIC REGRESSION MODEL
    u_model = LogisticRegression()
    u_model.fit(u_X_train, u_y_train)

    # PREDICT ON TEST SET
    u_y_pred = u_model.predict(u_X_test)

    # EVALUATE PERFORMANCE
    print("📌 ACCURACY SCORE:", accuracy_score(u_y_test, u_y_pred))
    print("📌 CLASSIFICATION REPORT:\n", classification_report(u_y_test, u_y_pred, target_names=["POOR", "RICH"]))

# RUN THE CLASSIFIER FUNCTION
u_train_rich_poor_classifier()
