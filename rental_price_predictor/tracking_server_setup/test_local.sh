docker run -it\
 -p 4000:80\
 -v "$(pwd):/home/app"\
 -e PORT=80\
 test-mlflow-tracking-setup mlflow server -p 80 --host 0.0.0.0