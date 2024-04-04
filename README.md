# HESSIAN: Health Evaluation and Surveillance for Sylvan Infirmities using Advanced Neural networks

## 💬 About the Project
Welcome to **HESSIAN**, a pioneering initiative led by "The Foresters 🌲" team, dedicated to the health evaluation and surveillance of plant life using cutting-edge analytical networking techniques. Our mission is to leverage advanced algorithms and data analytics to detect, classify, and understand the health of leaves, ultimately contributing to the preservation and care of our vital forest ecosystems.

## 🌲 Our Team: The Foresters
The Foresters 🌲 team is composed of three members, all in deep love with nature:
- Alexandre Eymaël (@alexandre-eymael)
- Badei Alrahel (@BadeiAlrahel)
- Louis Colson (@colson-louis)

## 🎓 Context
The HESSIAN project is carried out as part of the [INFO9023 - Machine Learning Systems Design](https://github.com/ThomasVrancken/info9023-mlops/tree/main) course at ULiège.

## 📝 Building Blocks
We provide a table detailing all the features implemented in this project, along with their respective implementation locations. Upon navigating to these locations, you will find an additional `README.md` file in the corresponding subdirectory, which serves to further explain and demonstrate the implemented feature.

|       #       |             Work package             | Required? | Implemented? |                                      Location                                     |
|-------------|------------------------------------|---------|:------------:|---------------------------------------------------------------------------------|
|    **1.1**    |              Pick a team             |     ✅     |       ✅      |                                 <a href="./">.<a>                                 |
|    **1.2**    |         Communication channel        |     ✅     |       ✅      |                                 <a href="./">.<a>                                 |
| **1.3** |          Use Case Selection         |     ✅     |       ✅      |                       <a href="./USECASE.md"> USECASE.md <a>                      |
| **1.4** |          Use Case Definition         |     ✅     |       ✅      |                       <a href="./USECASE.md"> USECASE.md <a>                      |
|       -       |                   -                  |     -     |       -      |                                         -                                         |
|    **1.5**    |  Setup a code versioning repository  |     ✅     |       ✅      |                                 <a href="./">.<a>                                 |
|    **1.6**    |              Find a name             |     ✅     |       ✅      |                                 <a href="./">.<a>                                 |
|       -       |                   -                  |     -     |       -      |                                         -                                         |
|    **2.1**    |       Exploratory Data Analysis      |     ✅     |       ✅      | <a href="./exploratory_data_analysis.ipynb"> exploratory_data_analysis.ipynb </a> |
| **2.2** |     Train your model    |     ✅     |       ✅      |                          <a href="./models"> models/ </a>                         |
| **2.3** |     Evaluate your model    |     ✅     |       ✅      |                          <a href="./models"> models/ </a>                         |
|    **2.4**    |          Weights and Biases          |     ❌     |       ✅      |                          <a href="./models"> models/ </a>                         |
|       -       |                   -                  |     -     |       -      |                                         -                                         |
|    **3.1**    |        API to serve the model        |     ✅     |       ✅      |                      <a href="./deployment"> deployment/ </a>                     |
|    **3.2**    |        Package API in a Docker       |     ✅     |       ✅      |                      <a href="./deployment"> deployment/ </a>                     |
|    **3.3**    |        Deploy API in the cloud       |     ✅     |       ✅      |                      <a href="./deployment"> deployment/ </a>                     |
|       -       |                   -                  |     -     |       -      |                                         -                                         |
|    **4.1**    |  Package model training in a Docker  |     ❌     |       ✅      |                      <a href="./deployment_train"> deployment_train/ </a>                                             |
|    **4.2**    | Run your model training in the cloud |     ❌     |       ✅      |                      <a href="./deployment_train"> deployment_train/ </a>                                                              |
|    **4.3**    |          Automated Pipeline          |     ❌     |       ❌      |                                                                                   |
|       -       |                   -                  |     -     |       -      |                                         -                                         |
| **5.1**       | Dashboard                            | ❌         | ✅            |                  <a href="./deployment"> deployment/ </a>                                                |
| **5.2**       | CICD                                 | ✅         | ✅            |                      <a href="./.github/workflows"> .github/workflows/ </a>                                                            |
| **5.3**       | CICD: Model Training                 | ❌         | ✅            |                      <a href="./.github/workflows/training.yml"> .github/workflows/training.yml </a>                                                             |
| **5.4**       | CICD: Model Deployment               | ❌         | ✅            |                      <a href="./.github/workflows/deploy.yml"> .github/workflows/deploy.yml </a>                                                             |
| **5.5**       | CICD: Pylint                         | ❌         | ✅            |                      <a href="./.github/workflows/pylint.yml"> .github/workflows/pylint.yml </a>                                                            |
| **5.6**       | CICD: Pytest                         | ❌         | ✅            |                      <a href="./.github/workflows/pytest.yml"> .github/workflows/pytest.yml </a>                                                             |

## 🌿 Gitflow Principles 
During the development of this project, we strictly adhered to Gitflow principles to maintain a structured and efficient workflow. Our main branch served as the repository's official release history, containing stable and approved code. Whenever we initiated work on a new feature, we created a dedicated branch branching off from the main branch. This allowed us to isolate development efforts and maintain a clean codebase.

Once a feature was fully developed and thoroughly tested, we initiated a pull request to merge the feature branch into the master branch. This integration process ensured that only completed and validated features were merged into the main branch, thereby preserving the stability and integrity of our codebase. By following this approach, we maintained a systematic and organized development cycle, enabling seamless collaboration among team members and facilitating the management of feature implementations. 

## 📃 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
