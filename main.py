""" Cameron Ela
    Description: This program conducts a Decision Tree
    Classification on a dataset about mushrooms.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# does all wrangling steps (cleaning and separating data)
def wrangle(df):
    new_df = df.dropna()
    new_df = new_df.drop_duplicates()
    y = new_df["class"]
    X = new_df.drop(columns="class")

    return X, y


def main():
    # read in csv and wrangle data
    mushrooms = pd.read_csv("mushrooms.csv")
    feature, target = wrangle(mushrooms)

    # add new mushroom and encodes each sample before removing new mushroom
    new_shroom = {
        "cap-shape": "x",
        "cap-surface": "s",
        "cap-color": "n",
        "bruises": "t",
        "odor": "y",
        "gill-attachment": "f",
        "gill-spacing": "c",
        "gill-size": "n",
        "gill-color": "k",
        "stalk-shape": "e",
        "stalk-root": "e",
        "stalk-surface-above-ring": "s",
        "stalk-surface-below-ring": "s",
        "stalk-color-above-ring": "w",
        "stalk-color-below-ring": "w",
        "veil-type": "p",
        "veil-color": "w",
        "ring-number": "o",
        "ring-type": "p",
        "spore-print-color": "r",
        "population": "s",
        "habitat": "u"
    }
    # Turn the given mushroom into a DataFrame
    new_shroom_df = pd.DataFrame([new_shroom])
    # concatenate the DataFrames and get dummy vars
    feature = pd.concat([feature, new_shroom_df], ignore_index=True)
    feature = pd.get_dummies(feature)   # given mushroom can be used for prediction
    new_shroom_df = feature.iloc[len(feature) - 1].copy()    # df for the prediction
    feature = feature.drop(feature.index[len(feature) - 1])    # drop given mushroom from original df
    X_train, X_test, y_train, y_test = train_test_split(feature, target,
    random_state=42, stratify=target)

    # conduct cross validation
    model = DecisionTreeClassifier()
    depth = [i for i in range(2, int(np.ceil((len(X_train) ** 0.5) + 1)))]
    hyp_param = {
        "criterion": ["entropy", "gini"],
        "max_depth": depth,
        "min_samples_split": [i for i in range(2, 21)],
        "min_samples_leaf": [i for i in range(1, 11)]
    }
    # conduct random search for best hyperparameters
    rscv = RandomizedSearchCV(estimator=model, param_distributions=hyp_param)
    rscv.fit(X_train, y_train)
    best = rscv.best_params_

    # train final model with optimized hyperparameters
    model = DecisionTreeClassifier(criterion=best["criterion"],
        max_depth=best["max_depth"], min_samples_split=best["min_samples_split"],
        min_samples_leaf=best["min_samples_leaf"])
    model.fit(feature, target)

    # create confusion confusion matrix
    y_pred = model.predict(X_test)
    mat = confusion_matrix(y_test, y_pred)
    mat_disp = ConfusionMatrixDisplay(confusion_matrix=mat)

    # create figure
    fig, ax = plt.subplots(1, 2, figsize=(20, 9))
    mat_disp.plot(ax=ax[0])
    ax[0].set(title="Mushroom Edibility Confusion Matrix")
    plot_tree(model, ax=ax[1], feature_names=feature.columns.tolist(),
        class_names=target.unique().tolist(), filled=True)
    ax[1].set(title="Decision Tree Visualization")
    fig.suptitle("Mushroom Decision Tree Results")
    fig.tight_layout()
    plt.savefig("mushrooms.png")

    # classify given mushroom
    pred_shroom = model.predict(new_shroom_df.values.reshape(1, -1))
    print("\nThe given mushroom has the predicted classification:", pred_shroom)


if __name__ == '__main__':
    main()
