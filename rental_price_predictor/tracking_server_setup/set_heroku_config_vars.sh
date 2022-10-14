APP_NAME="acsts-getaround-mlflow-server"

# Artifacts S3 folder location
heroku config:set ARTIFACT_ROOT=$ARTIFACT_ROOT -a $APP_NAME

# AWS credentials
heroku config:set AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -a $APP_NAME
heroku config:set AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -a $APP_NAME

# get DATABASE_URL from config vars
# (see https://devcenter.heroku.com/articles/connecting-to-heroku-postgres-databases-from-outside-of-heroku#don-t-copy-and-paste-credentials-to-a-separate-environment-or-app-code)
# and replace 'postgres' by 'postgresql' in it to be a valid BACKEND_STORE_URI to mlflow
# (see https://docs.sqlalchemy.org/en/14/core/engines.html#postgresql)
BACKEND_STORE_URI=$(heroku config:get DATABASE_URL -a ${APP_NAME} | sed 's/postgres/postgresql/')
heroku config:set BACKEND_STORE_URI=$BACKEND_STORE_URI -a $APP_NAME