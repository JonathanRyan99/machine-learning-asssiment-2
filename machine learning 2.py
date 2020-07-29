#https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/
import numpy as np # maffs
import pandas as pd # importing the data and some manipulation
import matplotlib.pyplot as plt # math visulization
import seaborn as sns #works of plt to make visulisation easier
import warnings #ignores common warnings user friendlyness
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split #allows me to easily split the data set (its also random)
from sklearn.neural_network import MLPClassifier# classifer model
from sklearn.metrics import accuracy_score # gives accurcy of the prediction
from sklearn.ensemble import RandomForestClassifier #use this to get random forest
from sklearn.model_selection import KFold

dataset = pd.read_csv('data_set.csv')
df = pd.DataFrame(dataset) #needed for kfold doesnt work without
#dataset.head(5) #test the import
emptyCell = dataset.isnull().sum() #checks for NULL data, none found in any of the coloums

#binarize the catigarical data
dataset = dataset.replace(to_replace = "Normal", value = 1) 
dataset = dataset.replace(to_replace = "Abnormal", value = 0)

#made this global for ease
X = df.iloc[:, 1:].values
Y = df.iloc[:,0].values


def section1(dataset):
   # plot1 = sns.boxplot( x=dataset["Status"], y=dataset["Vibration_sensor_1"] )
    #plt.savefig("boxplot.png")
    plot2 = sns.distplot(dataset['Vibration_sensor_2'])
    plt.savefig("density.png")
 
    
    
    

def datasplit(dataset, percentage):
    X = dataset.iloc[:, 1:12]
    Y = dataset.iloc[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size = percentage)
    print(" X_train, X_test, y_train, y_test: ",X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return(X_train, X_test, y_train, y_test)


def ann(X_train, X_test, y_train, y_test, epoch, hiddenlayer):
    #sets up the hidden layers, 2 layers of hiddenlayer nodes, epoch also given when function is called
    #learning rate of 0.01 (gives best accuracy = 0.86, once at 1000 epoch
    #solver = sgd which is stochastic gradient decnt which is linear -> logistic regression
    mlp = MLPClassifier(activation='logistic',hidden_layer_sizes=(hiddenlayer,hiddenlayer), max_iter= epoch, learning_rate_init = 0.01, solver= 'sgd')
    mlp.fit(X_train, y_train)
    MLP_predictions = mlp.predict(X_test)
    results = accuracy_score(y_test,MLP_predictions)
    return results

def tree(X_train, X_test, y_train, y_test, trees):
    #random forest
    #n_estimators = trees (decesion tree)
    #min_samples_leaf = needs at least n number of leaves
    #max_leaf_nodes = stops when it hits 50 leaves (defautl just goes till pure)
    clf = RandomForestClassifier(n_estimators = trees, min_samples_leaf= 5, max_leaf_nodes= 50)
    clf.fit(X_train, y_train)
    Forest_predictions = clf.predict(X_test)
    results = accuracy_score(y_test,Forest_predictions)
    return results

#function produces a grpah of accuracy against epoch size
def annGraph(epoch_limit,hiddenlayer):
    annEffciency = []
    for value in range(1, epoch_limit+1):  
        print("current epoch:",value,'/',epoch_limit)
        annEffciency.append(ann(X_train, X_test, y_train, y_test, value, hiddenlayer))
    
    plt.plot(annEffciency)
    plt.ylabel('accuracy value')
    plt.xlabel('epoch size')
    plt.savefig('ann_effciency.png')



#kfolds basically runs the model n_splits times each time generating a new split from your data set
#then calculate the accuracy of these and find the mean
#this should highlight any meaningful changes in accuracy as you change the parameters of the function
def annkfold():
    kfoldann = []
    kfoldsvar = KFold(n_splits = 10, shuffle= True)
    for trainIndex, testIndex in kfoldsvar.split(X, Y):
        kfoldann.append(ann(X[trainIndex], X[testIndex], Y[trainIndex], Y[testIndex], 200,))
     
    return np.mean(kfoldann)


def treekfold():
    kfoldtree= []
    kfoldsvar = KFold(n_splits = 2, shuffle= True)
    for trainIndex, testIndex in kfoldsvar.split(X, Y):
        kfoldtree.append(tree(X[trainIndex], X[testIndex], Y[trainIndex], Y[testIndex], 1000))
        
    return np.mean(kfoldtree)
    
    
#uncomment to use these

section1(dataset)
#X_train, X_test, y_train, y_test = datasplit(dataset, 0.10)
#annGraph(20,500)


    
#print("forest accuracy value: ",tree(X_train, X_test, y_train, y_test, 1000))
#print("ANN accuracy value: ",ann(X_train, X_test, y_train, y_test, 1000, 500))

# treesval=[10,50,100,1000,5000]
# length = len(treesval)
# results=[]
# for i in range(length):
#     results.append(tree(X_train, X_test, y_train, y_test, treesval[i]))
#     print(results)



#print("mean value of kfold ANN: ",annkfold())
#print("mean value for kfold tree", treekfold())

















