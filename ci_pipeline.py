
PROJECT_ID = "vertextesting-330105"
REGION = "us-central1" #though us-central is cheaper
PIPELINE_ROOT = "gs://vertextesting/pipeline_root"



import kfp
import requests
from typing import NamedTuple
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from google.cloud import aiplatform_v1beta1
from typing import Optional, Sequence, Dict, Tuple
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        HTML,
                        Input,
                        InputPath,
                        Model,
                        Output,
                        OutputPath,
                        Metrics,
                        ClassificationMetrics,
                        component
                        )

from kfp.v2 import compiler
from kfp.v2.google import experimental
from kfp.v2.google.client import AIPlatformClient



@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "psutil",
        "neptune-client"
    ],
    base_image="python:3.9",
    output_component_file="getting_dataset.yaml"
)
def get_data(
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
    transormed_data:Output[Dataset],
    EDA:Output[HTML]
):
    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split as tts
    import pandas as pd
    import neptune.new as neptune

    run = neptune.init(project='alekhya14.lavu/Iris', api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzIxYTRmOS1lODYxLTQzZGQtOWIxZi0wZjEyZDAzMDRkODEifQ==", run="IR-1")
    

    run["processed_dataset"].download(destination=transormed_data.path)
    run["EDA_report"].download(destination=EDA.path)
    run.stop()

    data = pd.read_csv(transormed_data.path)
    
    train, test = tts(data, test_size=0.3)
    train.to_csv(dataset_train.path)
    test.to_csv(dataset_test.path)



@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "xgboost",
        "joblib",
        "psutil",
        "neptune-client"
    ],
    base_image="python:3.9"
)
def train_xgb(
     dataset: Input[Dataset],
     metrics: Output[Metrics],
     model: Output[Model]
):
    
    from xgboost import XGBClassifier
    import xgboost as xgb
    import pandas as pd
    from joblib import dump
    import neptune.new as neptune
    
    data = pd.read_csv(dataset.path)
    
    run_1 = neptune.init(project='alekhya14.lavu/Iris', api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzIxYTRmOS1lODYxLTQzZGQtOWIxZi0wZjEyZDAzMDRkODEifQ==", run="IR-2")
    
    a=run_1["best/params"].fetch()

    xgmodel = XGBClassifier(
        objective="multi:softmax",
        use_label_encoder=False,
        gamma = a['gamma'],
        learning_rate = a['learning_rate'],
        max_depth = a['max_depth'],
        n_estimators = a['n_estimators']
    )
    
    xgmodel.fit(
        data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]).values,
        data.target.values,
    )

    xgmodel.save_model(model.path + ".bst")



@component(
    packages_to_install = [
        "pandas",
        "sklearn"
    ],
    base_image="python:3.9"
)
def grid_search(
     dataset: Input[Dataset],
     
) -> NamedTuple("Outputs", [("m", str),("n_n", int),("w", str)]):

    from sklearn.neighbors import KNeighborsClassifier
    from  sklearn.model_selection import GridSearchCV
    import pandas as pd
    
    data = pd.read_csv(dataset.path)
    print(data.columns)
    
    grid_params = {'n_neighbors' : [3,5,11,19], 'weights' : ['uniform','distance'],'metric':['euclidean','manhattan']}

    gs = GridSearchCV( KNeighborsClassifier(), grid_params, verbose=1, cv =3, n_jobs= -1)

    result = gs.fit(data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]).values,data.target.values)

    m = result.best_params_['metric']
    n_n = result.best_params_['n_neighbors']
    w = result.best_params_['weights']

    print(m,n_n,w)
    return(m,n_n,w)



@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "joblib"
    ],
    base_image="python:3.9"
)
def train_KNN(
     dataset: Input[Dataset],
     metrics: Output[Metrics],
     model: Output[Model],
     m : str,
     n_n : int,
     w : str
):
    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    from joblib import dump
    
    data = pd.read_csv(dataset.path)

    knn_model=KNeighborsClassifier(metric=m, n_neighbors=n_n, weights=w) 
    
    knn_model.fit(data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]).values,data.target.values)

    dump(knn_model, model.path + ".joblib")



@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "joblib"
    ],
    base_image="python:3.9"
)
def eval_nn(
    test_set: Input[Dataset],
    knn_model: Input[Model],
    metrics: Output[ClassificationMetrics],
    smetrics: Output[Metrics],
    model: Output[Model]
):

    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    from joblib import load, dump
    
    data = pd.read_csv(test_set.path)

    model_knn = load(knn_model.path + ".joblib")
    
    from sklearn.metrics import roc_curve
    
    y_scores =  model_knn.predict_proba(data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]))[:, 1]
    
    fpr, tpr, thresholds = roc_curve(
         y_true=data.target.to_numpy(), y_score=y_scores, pos_label=True
    )
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())
    
    from sklearn.metrics import confusion_matrix
    y_pred = model_knn.predict(data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]))
    
    metrics.log_confusion_matrix(
       ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
       confusion_matrix(
           data.target, y_pred
       ).tolist(),  # .tolist() to convert np array to list.
    )
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    preds = model_knn.predict(data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]))

    smetrics.log_metric("Precision KNN",precision_score(data['target'],preds, average='macro'))
    smetrics.log_metric("Recall KNN",recall_score(data['target'],preds, average='macro'))
    smetrics.log_metric("Accuracy KNN",accuracy_score(data['target'],preds))

    model.metadata["test_score"] = float(accuracy_score(data['target'],preds))

    dump(model_knn, model.path + ".joblib")



@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "xgboost"
    ],
    base_image="python:3.9"
)
def eval_xgboost(
    test_set: Input[Dataset],
    xgb_model: Input[Model],
    metrics: Output[ClassificationMetrics],
    smetrics: Output[Metrics],
    model: Output[Model]
):
    from xgboost import XGBClassifier
    import pandas as pd
    
    data = pd.read_csv(test_set.path)
    xgmodel = XGBClassifier()
    xgmodel.load_model(xgb_model.path + ".bst")
    
    from sklearn.metrics import roc_curve
    
    y_scores =  xgmodel.predict_proba(data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]))[:, 1]
    
    fpr, tpr, thresholds = roc_curve(
         y_true=data.target.to_numpy(), y_score=y_scores, pos_label=True
    )
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())
    
    from sklearn.metrics import confusion_matrix
    y_pred = xgmodel.predict(data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]))
    
    metrics.log_confusion_matrix(
       ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
       confusion_matrix(
           data.target, y_pred
       ).tolist(),  # .tolist() to convert np array to list.
    )
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    preds = xgmodel.predict(data.drop(columns=["target","Unnamed: 0","Unnamed: 0.1"]))

    smetrics.log_metric("Precision",precision_score(data['target'],preds, average='macro'))
    smetrics.log_metric("Recall",recall_score(data['target'],preds, average='macro'))
    smetrics.log_metric("Accuracy",accuracy_score(data['target'],preds))

    model.metadata["test_score"] = float(accuracy_score(data['target'],preds))

    xgmodel.save_model(model.path + ".bst")



@component(
    packages_to_install=["google-cloud-aiplatform", "joblib", "sklearn"],
    base_image="python:3.9"
)
def deploy_model(
    model: Input[Model],
    model_knn : Input[Model],
    project: str,
    region: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
    vertex_model_knn: Output[Model]
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    a=model.uri.replace("model", ""),
    b=model_uri_address = ''.join(a)
    print("Model_path"+b)
    #model.uri.replace("model", "model.joblib")

    #Uploading XGBoost model....
    deployed_model = aiplatform.Model.upload(
        display_name="XGBoost_Model",
        #artifact_uri = b,
        artifact_uri = model.uri.replace("model", ""),
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest"
        
    )

    main_endpoint= aiplatform.Endpoint.create(display_name="him",)
    
    endpoint = deployed_model.deploy(machine_type="n1-standard-2", endpoint=main_endpoint)

    endpoint.wait()
    
    ##Getting model endpoint id
    deployed_model_id = endpoint.list_models()[0].id

    #Uploading KNN model....
    deployed_model_knn = aiplatform.Model.upload(
        display_name="KNN_Model",
        artifact_uri = model_knn.uri.replace("model", ""),
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
    )
    endpoint_two = deployed_model_knn.deploy(machine_type="n1-standard-2", endpoint=main_endpoint, traffic_split={"0":60,deployed_model_id:40})
    endpoint_two.wait()
    # Save data to the output params
    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name
    vertex_model_knn.uri =deployed_model_knn.resource_name



@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to determine the pipeline Context.
    name="pipeline-test-1",
)
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION
):
    dataset_op = get_data()

    grid = grid_search(dataset_op.outputs["dataset_train"]).after(dataset_op)

    k_nn = train_KNN(
        dataset = dataset_op.outputs["dataset_train"],
        m = grid.outputs["m"],
        n_n = grid.outputs["n_n"],
        w = grid.outputs["w"] 
    ).after(grid)

    eval_knn = eval_nn(test_set=dataset_op.outputs["dataset_test"], knn_model=k_nn.outputs["model"]).after(k_nn)

    train_op = train_xgb(dataset_op.outputs["dataset_train"]).after(dataset_op)
    
    eval_op = eval_xgboost(
        test_set=dataset_op.outputs["dataset_test"],
        xgb_model=train_op.outputs["model"]
    ).after(train_op)
    
    #Deploying both model for canary deployment...
    deploy_task = deploy_model(
            model=eval_op.outputs["model"],
            model_knn = eval_knn.outputs['model'],
            project=project,
            region=region
        )
        
if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=pipeline, package_path='xgb_pipe.json')