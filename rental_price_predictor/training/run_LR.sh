APP_NAME="acsts-getaround-mlflow-server" # Replace with the name of the Heroku app that hosts the remote tracking server
IMAGE_NAME="getaround-mlflow-training" # Replace with the name of the Docker image to use for training job
MLFLOW_TRACKING_URI="https://acsts-getaround-mlflow-server.herokuapp.com/" # replace with your remote tracking server URL
PYTHON_SCRIPT_PATH="train_LR.py"
ARGS="" 

# If the Heroku Postgres DATABASE_URL has changed, update accordingly the BACKEND_STORE_URI in the app config vars
DATABASE_URL_POSTGRESQL=$(heroku config:get DATABASE_URL -a ${APP_NAME} | sed 's/postgres/postgresql/')
BACKEND_STORE_URI=$(heroku config:get BACKEND_STORE_URI -a ${APP_NAME})
if [ ! $BACKEND_STORE_URI = $DATABASE_URL_POSTGRESQL ]
then
    heroku config:set BACKEND_STORE_URI=$DATABASE_URL_POSTGRESQL -a $APP_NAME
fi

# Run the training job
docker run -it\
 -v "$(pwd):/home/app"\
 -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI\
 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID\
 -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY\
 $IMAGE_NAME python $PYTHON_SCRIPT_PATH $ARGS