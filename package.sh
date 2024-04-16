python3 setup.py sdist --formats=gztar
gsutil cp ./dist/hessian_trainer-0.1.tar.gz gs://hessian/pytorch-on-gcp/hessian/train/python_package/hessian_trainer-0.1.tar.gz