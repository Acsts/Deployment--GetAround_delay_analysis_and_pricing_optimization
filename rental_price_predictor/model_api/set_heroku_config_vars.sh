# This scripts set/updates each of Heroku app's config vars
# if it does not already correspond to the one stored in environment variable and if the latter is not empty
# (see https://devcenter.heroku.com/articles/connecting-to-heroku-postgres-databases-from-outside-of-heroku#don-t-copy-and-paste-credentials-to-a-separate-environment-or-app-code)

APP_NAME="acsts-getaround-price-predict"

# AWS credentials
if [[ ( ! $AWS_ACCESS_KEY_ID = $(heroku config:get AWS_ACCESS_KEY_ID -a ${APP_NAME}) ) && ( ! -z "$AWS_ACCESS_KEY_ID" ) ]]
then
    heroku config:set AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -a $APP_NAME
fi
if [[ ( ! $AWS_SECRET_ACCESS_KEY = $(heroku config:get AWS_SECRET_ACCESS_KEY -a ${APP_NAME}) ) && ( ! -z "$AWS_SECRET_ACCESS_KEY" ) ]]
then
    heroku config:set AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -a $APP_NAME
fi

# MLFlow tracking server URI
if [[ ( ! $MLFLOW_TRACKING_URI = $(heroku config:get MLFLOW_TRACKING_URI -a ${APP_NAME}) ) && ( ! -z "$MLFLOW_TRACKING_URI" ) ]]
then
    heroku config:set MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI -a $APP_NAME
fi