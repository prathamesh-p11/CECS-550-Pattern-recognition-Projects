import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def loadData():
    col_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    data = pd.read_csv('wine_quality.csv', skiprows=14, header=None, names=col_names)
    # print(data)
    return data


def dataAnalysis(data):
    pd.set_option('display.max_columns', None)  #to display all columns without truncation
    print(f"\nData statistics=> {data.describe()}\n");
    print(f"\nMissing Values => \n{data.isnull().sum()}\n");    #display all missing values
    print(f"\nClasses and their frequencies=> \n{data.iloc[:, -1].value_counts(dropna=False)}\n");
    plt.style.use('ggplot')
    pd.DataFrame.hist(data, figsize=[15, 30]);
    plt.subplots_adjust(hspace= 0.6)
    plt.show()

#cleaning data, removing anomalies
def DataPreprocessing(dataset):
    X = dataset.iloc[:, 0:11]
    y = dataset.iloc[:, -1]
    # checking data for anomalies
    if (np.where(np.isnan(X))):
        y.fillna(0, inplace=True)
    if (np.where(np.isnan(y))):
        y.fillna(0, inplace=True)

    # replacing all values > 10 with the max value less than or equal to 10
    j = []
    for i in range(0, len(y)):
        if (y[i] > 10):
            y[i] = 0
            j.append(i)

    for i in j:
        y[i] = y.max()

    counts = dataset.iloc[:, -1].value_counts(dropna=False)

    counts = sorted(counts.items(), key=lambda x: x[1])

    minClass = []
    for i in range(0, 2):
        minClass.append(counts[i][0])

    dataset.drop(dataset[dataset['quality'] == minClass[0]].index, inplace=True)
    print(f"\n(Quality)Classes and their frequencies after removing anomalies=> \n{dataset.iloc[:, -1].value_counts(dropna=False)}\n");
    X = dataset.iloc[:, 0:11]
    y = dataset.iloc[:, -1]
    return X, y


def splitData(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def showGraphs(data):
    colum_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                   'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    # Correlation matrix
    correlations = data.corr()
    # Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)
    # Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
    ax.set_xticklabels(
        colum_names,
        rotation=45,
        horizontalalignment='right'
    );
    ax.set_yticklabels(colum_names);
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(bottom = 0.2)
    sns.countplot(data["quality"], palette="muted")
    data["quality"].value_counts()
    plt.title('Quality classes and frequencies')
    plt.ylabel('Frequency')
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs fixed acidity')
    sns.barplot(x='quality', y='fixed acidity', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs volatile acidity')
    sns.barplot(x='quality', y='volatile acidity', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs citric acid')
    sns.barplot(x='quality', y='citric acid', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs residual sugar')
    sns.barplot(x='quality', y='residual sugar', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs chlorides')
    sns.barplot(x='quality', y='chlorides', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs free sulfur dioxide')
    sns.barplot(x='quality', y='free sulfur dioxide', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs total sulfur dioxide')
    sns.barplot(x='quality', y='total sulfur dioxide', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs density')
    sns.barplot(x='quality', y='density', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs pH')
    sns.barplot(x='quality', y='pH', data=data)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs sulphates')
    sns.barplot(x='quality', y='sulphates', data=data)
    plt.show()


    fig = plt.figure(figsize=(10, 6))
    plt.title('Quality vs alcohol')
    sns.barplot(x='quality', y='alcohol', data=data)
    plt.show()


def train_test(X_train, X_test, y_train, y_test, k):
    # training step
    from sklearn.neighbors import KNeighborsClassifier
    KNN_classifier = KNeighborsClassifier(n_neighbors=k)
    model = KNN_classifier.fit(X_train, y_train)
    model.fit(X_train, y_train)

    # testing step
    y_pred = KNN_classifier.predict(X_test)
    from sklearn import metrics
    print("\nKNN model trained and tested with Accuracy = ", (metrics.accuracy_score(y_test, y_pred)) * 100, "at k = ", k)


def crossValidate(data, X, y, c):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    print("\n\n\nCross validation with number of groups = ",c)
    KNN_classifier = KNeighborsClassifier(n_neighbors=5)
    # train model with cv = c
    cv_scores = cross_val_score(KNN_classifier, X, y, cv=c)
    print(f"\nCross validation scores => {cv_scores}")
    print(f"\ncv_scores mean:{np.mean(cv_scores)}")


def hyperTuningParam(data, X, y, T, R1, R2):
    from sklearn.neighbors import KNeighborsClassifier
    KNN_classifier = KNeighborsClassifier()
    print("\n\n\nHyper tuning parameters with number of groups for cross validation = ",T," and Range for optimizing k => [",R1,",",R2,"]")
    from sklearn.model_selection import GridSearchCV
    k_range = list(range(R1, R2))
    param_grid = dict(n_neighbors=k_range)
    knn_gscv = GridSearchCV(KNN_classifier, param_grid, cv=5)
    knn_gscv.fit(X, y)

    print(f"\n\nBest Param => {knn_gscv.best_params_} \n")
    print(f"\nBest Score => {knn_gscv.best_score_} \n")

def scaleFeatures(X):
    # feature scaling
    from sklearn.preprocessing import scale
    Xs = scale(X)
    return Xs

def main():
    #accept input as arguments
    k = int(sys.argv[1])
    T = int(sys.argv[2])
    R1 = int(sys.argv[3])
    R2 = int(sys.argv[4])

    data = loadData()
    dataAnalysis(data)
    X, y = DataPreprocessing(data)
    showGraphs(data)
    X = scaleFeatures(X)
    X_train, X_test, y_train, y_test = splitData(X, y)
    train_test(X_train, X_test, y_train, y_test, k)
    crossValidate(data, X, y, T)
    hyperTuningParam(data, X, y,T, R1, R2)

if __name__ == "__main__": main()