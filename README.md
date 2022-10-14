# MachinLearing-Classification

<h2>Dataset:
The Wisconsin Cancer Dataset</h2>
<br/>
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
<br/>

![image](https://user-images.githubusercontent.com/84762786/195777770-c1922789-4556-46bb-9dab-3e069c5b2143.png)

<h2>
Model type : &lt;DecisionTree_entropy, DecisionTree_gini, LogisticRegression, SVM>
</h2>
<p>
-Various data scaling methods and encoding methods
<br/>
-Various values of the model parameters for each model
<br/>
-Various values for the hyperparameters
<br/>
-Various numbers 𝑘for 𝑘 fold cross validation
<p>

<h3>
You can use AutoML function to automatically run different combinations of the above within a <br/>
“single major function”.
</h3>
def AutoML(scaler, encoder, model, label, target) :
    
    if not isinstance(train, pd.DataFrame):
        raise TypeError
    # if not isinstance(scaler, dict):
    #     raise TypeError
    
    predicted = {}

    print("Encoder Type : ",encoder)
    print("Scaler Type : ",scaler)
    print("Model Type : ",model)

    hyper_param = {
    "k-means": [5],
    "em": [5],
    "clarans": {"numCluster":[4], "numLocal":[5], "maxNeighbor":[5]},
    "dbscan": {"eps":[0.1], "minSample":[4]}

    for e in encoder:
        e_train = Encoding(train, e)
        print("------------------------------------")
        print("Encoder Type: ",e)
        for s in scaler:
            print("------------------------------------")
            print("Scaler Type : " , s)
            s_train = Scaling(e_train,s)
            for m in model:
                print("------------------------------------")
                print("Model Type : ",m)
                predicted = predict(m, label, target)


AutoML(scaler, encoder, model, df_label, df_target)       

<h2>
Result :
<h2/>
![image](https://user-images.githubusercontent.com/84762786/195777185-ece7a9d3-1f27-47b3-88af-5505f1532367.png)
