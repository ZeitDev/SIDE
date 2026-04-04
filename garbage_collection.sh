mlflow gc --tracking-uri ./mlruns
find /tmp -user $(whoami) -type f -mmin +60 -delete
