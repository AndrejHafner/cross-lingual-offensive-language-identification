import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

if __name__ == '__main__':
    df = pd.read_csv("../data/toxic-comment-classification/train.csv")
    df = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
    labels = df.columns
    count = np.sum(df.values, axis=0)

    print(np.sum(np.sum(df.values, axis=1) == 0))

    ax = sbn.barplot(x=labels, y=count, orient="vertical", order=["toxic", "obscene", "insult", "severe_toxic", "identity_hate", "threat"])
    ax.set(ylabel="Number of comments")
    plt.show()