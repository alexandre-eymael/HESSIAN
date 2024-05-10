# 🚀 Deployment Workflow
The file [deploy.yml](deploy.yml) implements the automated deployment procedure. This action is triggered when a pull request is merged on the main branch and automatically deploys our application and models on the VPS so that users can directly access the new features on [hessian.be](http://hessian.be).

## 🔒 Github Secrets
We use [Github Secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions) to safely store the credentials necessary to connect to the VPS. 

# 🐍 Pylint Workflow
The file [pylint.yml](pylint.yml) implements the automated Pylint procedure. This workflow is triggered by the deployment workflow before deploying anything on the VPS to check if the code complies with PEP8 guidelines. If the corresponding code does not achieve a minimum score of 8.0/10, this workflow and the following ones (e.g., the deployment) fail.

# 📄 Pytest Workflow
The file [pytest.yml](pytest.yml) implements the automated testing procedure. It is triggered once the deployment on the VPS is finished to ensure that the deployed API is working as expected. The corresponding tests can be found in the [tests](../../tests) directory.

# 🤖 Training Workflow
The file [training.yml](deploy.yml) implements the automated training procedure. When a commit name matches “!train-hessian!” it starts the training on Vertex AI.

## 🔒 Github Secrets
We used [Github Secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions) to safely store the google cloud credentials.