# GetAround deployment - Subproject : Car Rental Price Predictor

## Goal of the subproject

The goal of this subproject is to deploy in an app a Machine Learning model which predicts the daily rental price of a car.  
The implementation of this subproject is divided into 2 main steps :
1. Train different models on a remote MLFlow tracking server to pick the best one
2. Deploy an API for the best model to make its predictions on data submitted to the app 

## Result

- Online app hosting the MLFlow tracking server : **https://acsts-getaround-mlflow-server.herokuapp.com/**
- Online app hosting the model API for predictions : **https://acsts-getaround-price-predict.herokuapp.com/**

## Dataset

The dataset used for this project is provided by Jedha Bootcamp and is available in the 'training/input_data' subfolder of this subproject, or [here](https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv).

## API Usage

To get predictions from the car rental price predictor API, you can either:
- make a CURL request: see an example in the script `curl_api_tester.sh`,
- make a request using python: see an example in the script `python_api_tester.py`,
- or go directly to the API web interface: https://acsts-getaround-price-predict.herokuapp.com/

## Train your own models

If you want to train your own models, you can use the source code of this repository to build your own remote MLFlow tracking server and train your own models, by using the following steps. 

### Prerequisites - Installations

- Having read/write permissions to an AWS S3 bucket (to store the artifacts of your MLFlow tracking server)
- Install Docker: https://docs.docker.com/get-docker/
- Create a Heroku account: https://signup.heroku.com/login
- Install Heroku: https://devcenter.heroku.com/articles/heroku-cli

### I. Subproject folder

1. Clone this repository to create your project folder:
    ```sh
    git clone https://github.com/Acsts/Deployment--GetAround_delay_analysis_and_pricing_optimization.git
    ```

2. Place yourself in the 'rental_price_predictor' subproject directory:
    ```sh
    cd rental_price_predictor
    ```

### II. Setup remote tracking server

1. Place yourself in the 'tracking_server_setup' subdirectory:
    ```sh
    cd tracking_server_setup
    ```

2. Create the Heroku app hosting the server: 
    ```sh
    heroku create <YOUR_APP_NAME>
    ```

3. Create a free (with 'hobby-dev' plan) postgresql database hosted by Heroku, linked to your app: 
    ```sh
    heroku addons:create heroku-postgresql:hobby-dev -a <YOUR_APP_NAME>
    ```
    _Note: you can check that it has been correctly created: ```heroku addons -a <YOUR_APP_NAME>```_

4. Set your AWS S3 credentials and artifact root folder URI (the one in which you want to store artifacts of your MLFlow experiments) as environment variables. You can group them in a secrets script (for example here called 'secrets.sh') like this: 
    ```sh
    export ARTIFACT_ROOT=<YOUR_ARTIFACT_ROOT_URI>
    export AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
    export AWS_SECRET_ACCESS_KEY=<YOUT_AWS_SECRET_ACCESS_KEY>
    ```
    Then run 
    ```  
    source <PATH_TO_YOUR_SECRETS_SCRIPT (for example: secrets.sh)>
    ```

    **IMPORTANT: Never share your credentials or your secrets script.  
    (see https://docs.aws.amazon.com/general/latest/gr/aws-access-keys-best-practices.html#iam-user-access-keys).** 

5. Set all your app's necessary configuration variables:
    ```sh
    source set_heroku_config_vars.sh
    ```

6. Push the Docker image to the Heroku app and release it:

    ```sh 
    heroku container:login
    ```
    ```sh
    heroku container:push web -a <YOUR_APP_NAME>
    ```
    ```sh
    heroku container:release web -a <YOUR_APP_NAME>
    ```

=> Your remote tracking server is deployed, accessible at the URL _https://<YOUR_APP_NAME>.herokuapp.com_

### III. Train your models and monitor their performances

Once you have a remote tracking server running, you can train different type of models available by the script `train.py`. 

From the project root directory, place yourself in the 'training' subdirectory:
```sh
cd training
```
Then follow the steps below.

#### 1. Set your environment variables

- Set your remote tracking server'app BACKEND_STORE_URI config var as an environement variable (**do not share it**):
    ```sh
    export BACKEND_STORE_URI=$(heroku config:get DATABASE_URL -a <YOUR_APP_NAME>)
    ```
- Set your remote tracking server app URL as an environment variable called MLFLOW_TRACKING_URI:
    ```sh
    export MLFLOW_TRACKING_URI=https://<YOUR_APP_NAME>.herokuapp.com/
    ```

- Make sure your AWS credentials and ARTIFACT_ROOT defined in II.4 are still stored in your environment variables (if not, repeat step II.4), you will also need them for monitoring training.

#### 2. Train your models

To train your models in an minimal environment containing all the necessary tools and dependencies, first build a Docker image with the same name called in the MLProject file:
```sh
docker build . -t getaround-mlflow-training
```

Once the image is created, there are 2 ways to launch your training runs: 
- with a packaged MLProject, by running the mlflow command: 
    ```sh 
    mlflow run . -P <PARAM_1=VALUE_1> <...> -P <PARAM_N=VALUE_N>
    ```
    Example: ```mlflow run . -P regressor=Ridge -P alpha=0.5"``` runs a training job of linear regression with a alpha=0.5 ridge regularization.  

    Unfortunately this method does not allow to perform a grid search with several possible values by hyperparameter, as the `mlflow run` command does not yet implements multiple arguments handling for one flag (see https://github.com/mlflow/mlflow/issues/3743)

- by running the python command:
        ```sh
        source run.sh --<PARAM_1> <VALUES_LIST1> <...> --<PARAM_N> <VALUES_LIST_N>
        ```
        with each VALUES_LIST possibly constituted of values separated by spaces for GridSearch parameters, or a single value for other parameters.  
        Example: ```run.sh --regressor=RF --cv=3 --max_depth 20 50 --n_estimators 10 20``` runs a 3 cross-validations grid search of a random forest regressor, with max depths of 20 or 50 and number of trees of 10 or 20, and other parameters at their default value. 

### **List of available parameters in this train.py file:**  

The `regressor` parameter defines the model that will be trained, and its value can be chosen between:  
    - `LR` (default): Linear regression. no other parameter compatible, no grid search / cross-validation performed.    
    - `Ridge`: Linear regression with Ridge regularization. can be configured by setting `alpha` parameter. compatible with grid search / cross-validation, configurable by setting `cv` parameter.  
    - `RF`: Random forest regressor. can be configured by setting `n_estimators`, `max_depth`, `min_samples_split` and `min_samples_leaf`. compatible with grid search / cross-validation, configurable by setting `cv` parameter.  

The `cv` (integer) can be set in case of grid search / corss-validation to define the number of cross-validations. 

The other parameters (`alpha`, `max_depth`, etc.) define the behaviour of the models. Their purpose, types and default values are the same that those defined in the sklearn documentation for [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) and [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

#### 3. Visualize your trained jobs logs

You can access a,d compare all the runs you made previously on the remote tracking server app. 

## Source code

The source code is available in this repository and written in Python3.




