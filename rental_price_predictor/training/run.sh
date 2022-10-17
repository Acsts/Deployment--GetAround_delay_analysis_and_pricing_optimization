# Run the training job :
docker run -it\
 -v "$(pwd):/home/app"\
 -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI\
 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID\
 -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY\
 getaround-mlflow-training python train.py "$@"