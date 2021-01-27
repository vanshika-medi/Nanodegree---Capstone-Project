# Titanic Survivor Prediction using Microsoft Azure

This project is part of the Capstone Project for the Machine Learning Engineer Using Microsoft Azure Nanodegree Program. The dataset is the titanic dataset which was obtained from Kaggle. the aim of the project was to run an AutoML and Hyperdrive to select the best model and deploy it.


## Architecture

The flowchart provided below shows the steps being taken:

![Architecture](https://user-images.githubusercontent.com/68374253/105895430-85428c00-603b-11eb-99fe-d7ab4a85eb38.png)


## Dataset

Dataset used is the Titanic dataset which was obtained from [Kaggle](https://www.kaggle.com/c/titanic)
The aim is to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. The columns to determine this are Gender, age, size of the family, class of the passenger, Ticket, Fare etc.

### Access

To get the dataset in the Azure ML Studio, I first uploaded it in the environment in the same directory as the notebook and then used to pandas to read it. The code for that is given below:
```
data = pd.read_csv('titanic_dataset.csv')
```

## Automated ML

Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time consuming, iterative tasks of machine learning model development. It allows us to build ML models with high scale, efficiency, and productivity all while sustaining model quality.
The traditional manner of Machine Learning involves us to do all the things by hand and with much less efficiency. AutoML helps us to solve both problems.

Tasks done: 
* Import all the dependencies and register the dataset.
* We clean the data like filling null values, removing outliers etc.
* We then initiate the AutoMLConfig class which contains parameters like experiment_timeout_minutes (time duration for the experiment to run), task (regression/classification, in this case classification), label_column_name (target), training data, validation_data, compute target and primary_metric (in this case accuracy). In our program:
  * **experiment_timeout_minutes**: *30 minutes*: Anything longer than this would be rejected.
  * **task**: *Classification* :  The dataset required us to determine whether a person would survive or not hence making it a classification  program.
  * **primary_metric**: *Accuracy* : The accuracy tells that overall how often the model is making a correct prediction. We could also use other metrics like confusion matrix or F1-score however I dound Accuracy to give me a clear sense of the model performance.
  * **label_column_name**: *'Survived'* : The target column is the column that our model predicts. Hence in this case the target column is Survived.
  * **training_data** : *train_data*: It is the subdataset which was achieved by using train_test_split. The model trains according to this dataset.
  * **n-cross-validations** : *5*: To perform cross-validation, include the n_cross_validations parameter and set it to a value. This parameter sets how many cross validations to perform, based on the same number of folds. Hence metrics are calculated with the average of the five validation metrics.
  * **compute_target** : *compute_target*: In the notebook, we define compute target as the currently working compute cluster inatance defined by us which was computecluster1.
* Getting the best model and registering it. In this case the accuracy was 0.8127

### Results

* The models giving us the best accuracy are: VotingEnsemble and StackEnsemble.
* *VotingEnsemble* : It gave us an accuracy of 0.8217.
* *StackEnsemble* : It gave us an accuracy of 0.8034.
* Out of the 2, since VotingEnsemble gives a better accuracy, we register it as the best model.
* The parameters for it are: prefittedsoftvotingclassifier', min_samples_leaf=0.035789473684210524, min_samples_split=0.01, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=True, random_state=None, verbose=0, warm_start=False))],verbose=False))],flatten_transform=None, weights=[0.08333333333333333, 0.25,0.08333333333333333,  0.08333333333333333, 0.3333333333333333, 0.08333333333333333, 0.08333333333333333]))], verbose=False)

* AutoML Run Details:

![AutoML_RunDetails](https://user-images.githubusercontent.com/68374253/105891636-ddc35a80-6036-11eb-836e-a354dc7e01f9.png)

![AutoML_RunDetails1](https://user-images.githubusercontent.com/68374253/105891643-def48780-6036-11eb-9014-d6ab175b5937.png)

* AutoML Completed Run:

![AutoML_Completed](https://user-images.githubusercontent.com/68374253/105891627-dac86a00-6036-11eb-946e-7e15178f4f7a.png)

![AutoML_Completion](https://user-images.githubusercontent.com/68374253/105891630-dbf99700-6036-11eb-9249-9228869949fc.png)

* AutoML Best Models:

![AutoML_Top5Models](https://user-images.githubusercontent.com/68374253/105891647-e025b480-6036-11eb-95c2-092615324b2c.png)

![BestAutoMLModel](https://user-images.githubusercontent.com/68374253/105891652-e0be4b00-6036-11eb-8fa7-387882cc751f.png)

![AutoML_BestModel](https://user-images.githubusercontent.com/68374253/105891615-d8661000-6036-11eb-8206-3193363c5d23.png)

## Hyperdrive

Parameters which define the model architecture are known as hyperparameters and hence to find the best model for our data, we have to tune them. The following parameters are the ones we used in this project:

* **Parameter Sampler** : *BayesianParameterSampling* is being used. It defines Bayesian sampling over a hyperparameter search space.
* **Estimator** : An estimator needs to be defined with some sample hyperparameters. The SKLearn estimator for Scikit-Learn model training requires us to input the arguments like the source directory of the file, the name of the training file as well as the compute target being used.
* **HyperDriveConfig** : The HyperDriveConfig is where all the parameters for hyperdrive are set. It incles the above mentioned parameter sampler, early termination policy, estimator along with primary metrics being used, total_runs and max_concurrent_runs. We then submit this hyperdrive_config, retrieve the best possible model and register it.

* We imported dependencies and initialize an experiment.
* We then submit the run and et the RunDetails.
* We check the best model and compare it with the best model form the AutoML run. In this case, the accuracy of Hyperdrive is 0.832 dch is more than the accuracy of the AutoML run hence we decided to deploy it.

### Results

* The best model used Logistic Regression with an accuracy of 0.8324
* Hyperdrive RunDetails:

![Hyperdrive_RunDetails](https://user-images.githubusercontent.com/68374253/105893206-d00ed480-6038-11eb-9485-c4f4ef113fa1.png)

![Hyperdrive_RunDetails1](https://user-images.githubusercontent.com/68374253/105893207-d0a76b00-6038-11eb-9719-92c8c3d29a0f.png)

* Hyperparameter Configuration:

![Hyperparameter_configuration](https://user-images.githubusercontent.com/68374253/105893209-d1400180-6038-11eb-8c73-91750594f6db.png)

![Hyperparameter_configuration1](https://user-images.githubusercontent.com/68374253/105893210-d1d89800-6038-11eb-9563-fce71a19aec8.png)

* Hyperdrive Completion:

![Hyperdrive_Completed](https://user-images.githubusercontent.com/68374253/105893193-cd13e400-6038-11eb-9717-507a945cd179.png)

![Hyperdrive_Completed1](https://user-images.githubusercontent.com/68374253/105893198-ce451100-6038-11eb-9254-6c136abe5527.png)

![Hyperdrive_completion](https://user-images.githubusercontent.com/68374253/105893199-cedda780-6038-11eb-9e69-4374b7fd309e.png)

* Hyperdrive Best Model:

![Hyperdrive_BestModelSaved](https://user-images.githubusercontent.com/68374253/105893182-cb4a2080-6038-11eb-9264-0c00bb64d288.png)


## Model Deployment: 

*  We import conda dependencies and get 'scoring-uri' and get the logs.
* Next isnrun endpoint.py to check if it is deployed.

* Deployment:

![Hyperdrive_Deployment](https://user-images.githubusercontent.com/68374253/105893204-cf763e00-6038-11eb-9c25-100f1b7b80d2.png)

![Healthy_deployed_status](https://user-images.githubusercontent.com/68374253/105893177-ca18f380-6038-11eb-9468-e83d864ee9a0.png)

![Healthy_deployed_status2](https://user-images.githubusercontent.com/68374253/105893180-cab18a00-6038-11eb-9664-b49910ea1374.png)

![Deploy_consume_details](https://user-images.githubusercontent.com/68374253/105893167-c5ecd600-6038-11eb-9b62-c88af147ee80.png)

![Logs](https://user-images.githubusercontent.com/68374253/105893214-d309c500-6038-11eb-90b5-1e34edcf362d.png)

![Obtaining_scoring_uri](https://user-images.githubusercontent.com/68374253/105893219-d3a25b80-6038-11eb-8c73-be3d240d13ba.png)

![Prediction_endpoint](https://user-images.githubusercontent.com/68374253/105893222-d43af200-6038-11eb-9bd1-497c3dd00039.png)

* Deleting service:

![Deleted_web_service](https://user-images.githubusercontent.com/68374253/105893163-c5543f80-6038-11eb-96b1-a2a81c5857f8.png)

* Experiments and Endpoint:

![Endpoint_running](https://user-images.githubusercontent.com/68374253/105893171-c8e7c680-6038-11eb-9c6e-7229f9bde5a2.png)

![Experiments](https://user-images.githubusercontent.com/68374253/105893176-c9805d00-6038-11eb-87da-f80717f128d3.png)


## Screen Recording

The screencast for the project is here: [Screencast](https://youtu.be/TL0_Kg9Vo5s)

## Standout Suggestions

* We could convert our model to ONNX format. (There were difficulties in doing so.)
* Enable logging in a deployed app.
* To compare, I uploaded the predictions using traditional ML method.
