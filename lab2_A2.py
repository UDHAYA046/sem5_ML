import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def u_train_rich_poor_classifier():
    """
    LOAD DATA, LABEL AS RICH OR POOR, TRAIN CLASSIFIER,
    RETURN CLASSIFICATION REPORT AND CONFUSION MATRIX
    """

    # LOAD THE EXCEL FILE i.e. PURCHASE BEHAVIOR OF CUSTOMERS
    u_excel_data = pd.read_excel(
        r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx",
        sheet_name="Purchase data"
    )

    # CREATE TARGET LABEL i.e. RICH = 1 IF PAYMENT > 200, ELSE POOR = 0
    u_excel_data["u_label_class"] = u_excel_data["Payment (Rs)"].apply(lambda x: 1 if x > 200 else 0)

    # FEATURES i.e. CANDIES, MANGOES, MILK QUANTITIES
    u_features_X = u_excel_data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()

    # TARGET LABEL
    u_target_y = u_excel_data["u_label_class"].to_numpy()

    # TRAIN LOGISTIC REGRESSION ON FULL DATA (SMALL DATASET)
    u_model = LogisticRegression()
    u_model.fit(u_features_X, u_target_y)

    # PREDICT ON SAME DATA (DEMO PURPOSE ONLY)
    u_predictions = u_model.predict(u_features_X)

    # RETURN REPORT AND CONFUSION MATRIX
    u_report = classification_report(u_target_y, u_predictions, target_names=["POOR", "RICH"], zero_division=0)
    u_conf_matrix = confusion_matrix(u_target_y, u_predictions)

    return u_report, u_conf_matrix

if __name__ == "__main__":
    u_report_text, u_matrix = u_train_rich_poor_classifier()

    print(" CLASSIFICATION REPORT:")
    print(u_report_text)

    sns.heatmap(u_matrix, annot=True, cmap="Blues", fmt='d',
                xticklabels=["Predicted POOR", "Predicted RICH"],
                yticklabels=["Actual POOR", "Actual RICH"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
