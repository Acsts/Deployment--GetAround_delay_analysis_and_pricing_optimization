# This scripts set/updates each of Heroku app's config vars
# if it does not already correspond to the one stored in environment variable and if the latter is not empty
# (see https://devcenter.heroku.com/articles/connecting-to-heroku-postgres-databases-from-outside-of-heroku#don-t-copy-and-paste-credentials-to-a-separate-environment-or-app-code)

APP_NAME="acsts-getaround-mlflow-server"

# Update artifacts S3 folder location
if [[ ( ! $ARTIFACT_ROOT = $(heroku config:get ARTIFACT_ROOT -a ${APP_NAME}) ) && ( ! -z "$ARTIFACT_ROOT" ) ]]
then
    heroku config:set ARTIFACT_ROOT=$ARTIFACT_ROOT -a $APP_NAME
fi

# AWS credentials
if [[ ( ! $AWS_ACCESS_KEY_ID = $(heroku config:get AWS_ACCESS_KEY_ID -a ${APP_NAME}) ) && ( ! -z "$AWS_ACCESS_KEY_ID" ) ]]
then
    heroku config:set AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -a $APP_NAME
fi
if [[ ( ! $AWS_SECRET_ACCESS_KEY = $(heroku config:get AWS_SECRET_ACCESS_KEY -a ${APP_NAME}) ) && ( ! -z "$AWS_SECRET_ACCESS_KEY" ) ]]
then
    heroku config:set AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -a $APP_NAME
fi

# get DATABASE_URL from config vars and replace 'postgres' by 'postgresql' in it 
# to be a valid BACKEND_STORE_URI to mlflow (see https://docs.sqlalchemy.org/en/14/core/engines.html#postgresql)
DATABASE_URL_POSTGRESQL=$(heroku config:get DATABASE_URL -a ${APP_NAME} | sed 's/postgres/postgresql/')
BACKEND_STORE_URI=$(heroku config:get BACKEND_STORE_URI -a ${APP_NAME})
if [ ! "$BACKEND_STORE_URI" = $DATABASE_URL_POSTGRESQL ]
then
    heroku config:set BACKEND_STORE_URI=$DATABASE_URL_POSTGRESQL -a $APP_NAME
fi