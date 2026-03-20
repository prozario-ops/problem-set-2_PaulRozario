'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as lr

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """

    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    

    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()
def run_part5():
    print("Loading Part 5 data...")

    df_train = pd.read_csv("data/df_arrests_train.csv")
    df_test = pd.read_csv("data/df_arrests_test.csv")

    features = ['current_charge_felony', 'num_fel_arrests_last_year']

    X_train = df_train[features].fillna(0)
    y_train = df_train['y']

    X_test = df_test[features].fillna(0)
    y_test = df_test['y']

    # Logistic Regression

    print("\nLogistic Regression ...")

    lr_model = lr(C=0.1, max_iter=1000)
    lr_model.fit(X_train, y_train)

    prob_lr = lr_model.predict_proba(X_test)[:, 1]



    # Decision Tree

    print("\n Decision Tree...")
    dt_model = DTC(max_depth=1, random_state=120)
    dt_model.fit(X_train, y_train)

    prob_dt = dt_model.predict_proba(X_test)[:, 1]


    # Calibration Plots

    print("\nLogistic Regression Calibration Plot")
    calibration_plot(y_test, prob_lr, n_bins=5)
    

    print("Decision Tree Calibration Plot")
    calibration_plot(y_test, prob_dt, n_bins=5)

    # Which model is more calibrated

    print("\nWhich model is more calibrated?")
    print("\n Logistic regression "
    "is slightly more calibrated since it gives smoother probability estimates, " \
    "while the decision tree tends to be more rigid.Both models look almost the same on the calibration plots. " \
    "This is because most of the predicted probabilities are very low, " \
    "so everything falls into only a couple of bins.  Overall, " \
    "there is not a noticeable difference between them.")

