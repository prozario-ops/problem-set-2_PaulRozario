'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl as etl
import part2_preprocessing as preprocessing
import part3_logistic_regression as logistic_regression
import part4_decision_tree as decision_tree
import part5_calibration_plot as calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    part1=etl
    if part1:
        print("Part 1 ETL completed successfully and datasets saved in `./data/`")
    else:
        print("Part 1 ETL failed.")
    # PART 2: Call functions/instanciate objects from preprocessing
    part2=preprocessing
    if part2:
        print("Part 2 Preprocessing completed successfully and `df_arrests` saved in `./data/`")
    else:
        print("Part 2 Preprocessing failed.")
    # PART 3: Call functions/instanciate objects from logistic_regression
    part3=logistic_regression
    if part3:
        print("Part 3 Logistic Regression completed successfully and model results saved in `./data/`")
    else:
        print("Part 3 Logistic Regression failed.")
    # PART 4: Call functions/instanciate objects from decision_tree

    # PART 5: Call functions/instanciate objects from calibration_plot


if __name__ == "__main__":
    main()