# Machine Learning Canvas

## Background
- `users`: We target farmers, botanists, agricultural scientists, environmentalists, nature enthusiasts.
- `goals`: Enable our users to determine whether a leaf is sick and identify the specific disease it has without any prior botanical knowledge.
- `pains`: Determining the health status of a leaf often requires botanical knowledge, which might be time-consuming to acquire.

## Value Proposition
- `product`: We propose an application for automatically identifying the health status and potential disease of a leaf based on a picture.
- `alleviates`: Users do not need to have any specific botanical knowledge.
- `advantages`: Users have an accurate and straightfoward way of identifying the health status and potential disease of a leaf.

## Objectives
1. Provide the user with a responsible API and/or interface to use our product
2. Classify a leaf based on a picture of it, using the model that is the most suited to the user's needs

## Solution
- `core features`: 
        - Predict the specific illness afflicting the leaf.
        - User feedback process for incorrectly classified illness.
- `integration`:
        - The model will run on a VPS as well as its API to interact with it.
        - The API will be used to create a user-friendly web interface for uploading and predicting on user images.
- `alternatives`:
        - Allow users to add content manually and classify them.
- `constraints`:
        - Maintain low latency when predicting user inputs.
        - Complete category of illnesses so that the model can predict any of them.
- `out-of-scope`:
        - More sophisticated model to increase the accuracy (More: ram usage & inference latency).

## Feasibility
- `data`: We have a [dataset](https://www.kaggle.com/datasets/nirmalsankalana/plant-diseases-training-dataset) composed of ~40,000 annotated images
- `team`: We are three experienced data scientists
- `infrastructure`: 
    - `local`: We will use OVH hosting services to deploy our API and serve our model (using Flask) by ourselves.
    - `cloud`: We will deploy our model in the cloud as well.

## Data
- Kaggle [dataset](https://www.kaggle.com/datasets/nirmalsankalana/plant-diseases-training-dataset/data) composed of ~40,000 annotated leaves images with 63 different leaf disease classes. We will devide the data into trainnig and testing sets with 80% and 20% respectively for each class we have.

## Metrics
- Binary precision and recall, i.e., healthy from unhealthy leaf detection.
- Per-class precision and recall.
- Our focus will be on recall and especially for binary classification since we beleive that the cost of missing a sick leaf is very high.

## Evaluation
- ?

## Modeling 
- Deep learning model using convolutional neural networks.

## Inference
- We opt for online inference method, i.e. handling real-time requests, where a user can interact with our API and/or interface online.

## Feedback
- Allow users to report issues related to misclassification.
- Allow users to upload images with their corresponding illnesses.

## Project
- Cf. [Project Statement](https://github.com/ThomasVrancken/info9023-mlops/blob/main/project/project_description.pdf)