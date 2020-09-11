import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------------------
# -------------------------------------- DATA PROCESSING -------------------------------------------
# --------------------------------------------------------------------------------------------------

def loadData():
    col_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                 'Marginal Adhesion',
                 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=col_names)
    print("\n\n-----------------------------------------------------------------------------------------------------")
    print(f"Data head => \n{data.head}")
    print("------------------------------------------------------------------------------------------------------\n")
    return data, col_names


# cleaning data, removing anomalies
def DataPreprocessing(dataset):
    col_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                 'Marginal Adhesion',
                 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    X = dataset.iloc[:, 0:10]
    y = dataset.iloc[:, -1]

    print('\n---------------------------------- MISSING VALUES -----------------------------------------')
    # According to the description of the data the missing values are encoded as '?' in the original data
    dataset = dataset.replace('?', np.NaN)
    print('Number of missing values:')
    for col in dataset.columns:
        print('\t%s: %d' % (col, dataset[col].isna().sum()))

    # the missing values in the columns are replaced by the median value of that column. since we observed that some
    # columns hav outliers, and mean is heavily affected by outliers, we fill them with medians
    dataset = dataset.fillna(dataset.median())
    print('\n\nNumber of missing values after filling with medians:')
    for col in dataset.columns:
        print('\t%s: %d' % (col, dataset[col].isna().sum()))

    print('\n---------------------------------- Datatypes -----------------------------------------')

    # all columns are int except 'Bare Nuclei' is String - to represent '?'. We must first convert the column
    # into numric values first
    print(f"\nBefore converting data type of Bare Nuclei: \n {dataset.dtypes}")
    dataset['Bare Nuclei'] = dataset['Bare Nuclei'].astype('int64')
    print(f"\nAfter converting data type: \n {dataset.dtypes}")

    print('\n---------------------------------- OUTLIERS -----------------------------------------')
    # we do not need sample code number and it would create irrelevant boxplot size scaling problems
    dataset_plot = dataset.drop(['Sample code number'], axis=1)
    dataset_plot.boxplot(figsize=(15, 10))
    plt.title('Outliers', loc='center')
    plt.xticks(size=7)  # reduce size of label text
    plt.show()  # only 5 columns have abnormally high values (Marginal Adhesion, Single Epithetial Cell Size,
    # Bland Cromatin, Normal Nucleoli, and Mitoses)

    # removing outliers.
    # we can compute the z-score for each attribute. Any z-score greater than 3 or less than -3 is
    # considered to be an outlier and we remove these instances
    from scipy import stats
    z = np.abs(stats.zscore(dataset))
    print(f"\n\nZ-Score => \n{z}")
    pd.set_option('display.max_columns', None)  # to display all columns without truncation

    print(f"\nBefore removing outliers dataset = [{dataset.shape}]")
    dataset = dataset[(z < 3).all(axis=1)]
    print(f"\nAfter removing outliers dataset = [{dataset.shape}]")

    '''
    # ----------------------------Frequency distribution----------------------------
    Columns = [col for col in dataset.columns[0:9]]
    for i in range(len(Columns)):
        sns.FacetGrid(dataset, hue="Class", height=5, aspect=2, margin_titles=True).map(sns.kdeplot, Columns[i],
                                                                                        shade=True).add_legend()
        plt.title(f"Frequency distribution of {col_names[i + 1]}")
        plt.show()
    '''

    X = dataset.iloc[:, 0:9]
    y = dataset.iloc[:, -1]

    return dataset,X,y


def featureAnalysis(data, col_names):
    # ----------------------HEAT MAP----------------
    data = data.drop(['Sample code number'], axis=1)
    correlations = data.corr()          #based on Pearson Coefficient of Correlation
    # Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)

    # Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
    ax.set_xticklabels(
        col_names[1:11],
        rotation=45,
        horizontalalignment='right'
    );
    ax.set_yticklabels(col_names[1:11]);
    plt.title("Feature correlation heatmap")
    plt.show()

    # -----------------------SelectKBest------------------------
    X = data.iloc[:, 0:9]   #features
    y = data.iloc[:, -1]    #target variable : Class


    from sklearn.feature_selection import SelectKBest, chi2
    selector = SelectKBest(chi2, k=5)
    selector.fit(X, y)
    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices=True)
    features_df_new = X.iloc[:, cols]
    print(f"\n\n\n------------------------------ Selecting features ------------------------------------------")
    print(f"Best 5 features using SelectKBest= > \n {features_df_new}")
    # we may get different top feature results from heatmap and slectKBest because one uses Pearson Coefficient of
    # Correlation and other uses chi^2

def split_data(X,y):

    from sklearn.model_selection import train_test_split
    #Stratify makes proportional splits
    #ex if y is a binary var with values 0 and 1 and there are 25% 0s and 75% 1s, then stratify will make sure that
    #random split has 25% 0s and 75% 1s
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def scaleFeatures(X):
    # feature scaling
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    return Xs

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------
# -------------------------------Principal Component Analysis---------------------------------------
# --------------------------------------------------------------------------------------------------
def pca(X, data):
    print("\n\n---------------------------------- Principal Component Analysis ------------------------------------------ ")
    data = data.drop(['Sample code number'], axis=1)

    from sklearn.decomposition import PCA
    print(f"\n\nX = {X.shape}")
    print(f"\n\nD = {data}")

    pca_var = PCA()
    pca_var.fit(X)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(pca_var.explained_variance_ratio_.cumsum(), 'bo-', markersize=8)
    plt.title("Elbow Curve for Principal Component Analysis")
    plt.ylabel('Explained Variance')
    plt.xlabel('Component Number')
    sns.set_style("whitegrid")
    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalComponentsDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    pcaDF = pd.concat([principalComponentsDf, data[['Class']]], axis=1)
    print(f"\nVARIANCE : {pca.explained_variance_ratio_.cumsum()}")

    pcaDF = pcaDF.dropna()          #since we are fitting PCA on the trainig set only, we remove all extra rows having NaN values because of using whole dataset
    print(f"\n PCA DF =>\n{pcaDF}\n")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [2,4]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = pcaDF['Class'] == target
        ax.scatter(pcaDF.loc[indicesToKeep, 'principal component 1']
                   , pcaDF.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# -----------------------------------K Nearest Neighbours-------------------------------------------
# --------------------------------------------------------------------------------------------------

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



# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# --------------------------------- Artificial Neural Network --------------------------------------
# --------------------------------------------------------------------------------------------------

class ANN():
    np.random.seed(2)

    def __init__(self, layers=[9, 4, 1], lr=0.001, epochs=200):
        self.layers = layers
        self.lr = lr
        self.wb = {}
        self.cost = []
        self.epochs = epochs
        self.X = None
        self.y = None

    def _init_weights(self):
        # initialize weights and biases
        self.wb['W1'] = np.random.randn(self.layers[0], self.layers[1])
        self.wb['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.wb['b1'] = np.random.randn(self.layers[1], )
        self.wb['b2'] = np.random.randn(self.layers[2], )

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _sigmoid(self, Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def _dRelu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    # binary cross entropy loss
    def _entropy_loss(self, Y, Yhat):
        nsample = len(Y)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(Yhat), Y) + np.multiply((1 - Y), np.log(1 - Yhat))))
        loss = np.squeeze(loss)
        return loss

    def _forward_prop(self):
        # forward propagation
        Z1 = self.X.dot(self.wb['W1']) + self.wb['b1']
        A1 = self._relu(Z1)
        Z2 = A1.dot(self.wb['W2']) + self.wb['b2']
        yhat = self._sigmoid(Z2)
        cost = self._entropy_loss(self.y, yhat)

        # save parameters
        self.wb['Z1'] = Z1
        self.wb['A1'] = A1
        self.wb['Z2'] = Z2

        return yhat, cost

    def _backward_prop(self, yhat):
        # Backpropagation
        dl_wrt_yhat = -(np.divide(self.y, yhat) - np.divide((1 - self.y), (1 - yhat)))
        dl_wrt_sig = yhat * (1 - yhat)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.wb['W2'].T)
        dl_wrt_w2 = self.wb['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * self._dRelu(self.wb['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        # updates
        self.wb['W1'] = self.wb['W1'] - self.lr * dl_wrt_w1
        self.wb['W2'] = self.wb['W2'] - self.lr * dl_wrt_w2
        self.wb['b1'] = self.wb['b1'] - self.lr * dl_wrt_b1
        self.wb['b2'] = self.wb['b2'] - self.lr * dl_wrt_b2

    def fit(self, X, y):
        self.X = X
        self.y = y
        self._init_weights()

        for i in range(self.epochs):
            yhat, loss = self._forward_prop()
            self.cost.append(loss)
            self._backward_prop(yhat)

    def predict(self, X):
        Z1 = X.dot(self.wb['W1']) + self.wb['b1']
        A1 = self._relu(Z1)
        Z2 = A1.dot(self.wb['W2']) + self.wb['b2']
        pred = self._sigmoid(Z2)
        return np.round(pred)

    def plot_loss(self):
        plt.plot(self.cost)
        plt.title(f"Loss curve for {self.epochs} epochs")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()

    def accuracy(self, y, yhat):
        return int(sum((y == yhat)) / len(y) * 100)
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------



def main():
    pd.set_option('display.max_columns', None)  # to display all columns without truncation
    data, col_names = loadData()
    data,X,y = DataPreprocessing(data)
    featureAnalysis(data, col_names)
    X = scaleFeatures(X)

    X_PCA = X
    data_PCA = data

    inputs = X
    outputs = y

    '''
    print(f"X b4 PCA: {X.shape}")
    print(f"y b4 PCA: {y.shape}")
    print(f"data b4 PCA: {data.shape}")
    pca(X, data)
    print(f"X af PCA: {X.shape}")
    print(f"y af PCA: {y.shape}")
    print(f"data af PCA: {data.shape}")
    '''
    print("\n\n\n--------------------------- Classification model accuracy comparison --------------------------------")
    # ------------------------------------- KNN ----------------------------------------
    import math
    T = 5
    R1 = 1
    R2 = 25

    hyperTuningParam(data, X, y, T, R1, R2)
    X_train, X_test, y_train, y_test = split_data(inputs, outputs)
    train_test(X_train, X_test, y_train, y_test, 12)

    # ------------------------------ Neural Network ------------------------------------
    data = data.drop(['Sample code number'], axis=1)

    data['Class'] = data['Class'].map({2: 0, 4: 1})
    outputs = data.Class.values.reshape(outputs.shape[0], 1)  # IMPORTANT: RESHAPE TARGET TO 2D ARRAY
    X_train, X_test, y_train, y_test = split_data(inputs, outputs)

    ann = ANN(layers=[9,8,1])
    ann.fit(X_train, y_train)
    pred = ann.predict(X_test)
    print(f"Neural Network Accuracy = {ann.accuracy(pred, y_test)}")
    ann.plot_loss()

    # -------------------------- Principal Component Analysis --------------------------
    pca(X_PCA, data_PCA)


if __name__ == "__main__": main()