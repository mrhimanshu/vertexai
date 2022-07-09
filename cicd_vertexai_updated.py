
# %%
PROJECT_ID = "vertextesting-355712"
REGION = "us-central1" #though us-central is cheaper
# through gstutil we access storage
PIPELINE_ROOT = "gs://vertextestingpipeline/pipeline_root"

# %%
from typing import NamedTuple
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component)

from kfp.v2 import compiler
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

# %%
@component(
    packages_to_install=["pandas","sklearn"],
    base_image="python:3.9",
    output_component_file="getting_data.yaml"
)
def get_data(
    train_dataset : Output[Dataset],
    test_dataset : Output[Dataset]
):

    from sklearn import datasets
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data_raw = datasets.load_breast_cancer()
    data = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
    data['target'] = data_raw.target

    train, test = train_test_split(data, test_size=0.3)

    train.to_csv(train_dataset.path)
    test.to_csv(test_dataset.path)


# %%
@component(
    packages_to_install=["pandas","sklearn","xgboost"],
    base_image="python:3.9",
    output_component_file="training.yaml"
)
def training(
    train_data : Input[Dataset],
    test_data : Input[Dataset],
    model : Output[Model],
    metrics :Output[ClassificationMetrics],
    smetrics: Output[Metrics],
    threshold_value_str : str
) -> NamedTuple("Outputs", [("accuracy", str)]):

    from xgboost import XGBClassifier
    import pandas as pd
    import json

    data = pd.read_csv(train_data.path)

    model_xg = XGBClassifier(
        objective="binary:logistic"
    )
    model_xg.fit(
        data.drop(columns=["target"]),
        data.target,
    )

    score = model_xg.score(
        data.drop(columns=["target"]),
        data.target,
    )

    model.metadata["train_score"] = float(score)
    model.metadata["framework"] = "XGBoost"

    model_xg.save_model(model.path+".bst")

    from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, confusion_matrix
    
    data_test = pd.read_csv(test_data.path)

    y_scores =  model_xg.predict_proba(data_test.drop(columns=["target"]))[:, 1]
    fpr, tpr, thresholds = roc_curve(
         y_true=data_test.target.to_numpy(), y_score=y_scores, pos_label=True
    )
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())
    
    y_pred = model_xg.predict(data_test.drop(columns=["target"]))
    
    metrics.log_confusion_matrix(
       ["False", "True"],
       confusion_matrix(
           data_test.target, y_pred
       ).tolist(),  # .tolist() to convert np array to list.
    )

    accuracy = accuracy_score(data_test['target'],y_pred)
    smetrics.log_metric("Accuracy",accuracy)

    def threshold_check(val1, val2):
        cond = "false"
        if val1 >= val2 :
            cond = "true"
        return cond
        
    threshold_value = json.loads(threshold_value_str)
    deploy = threshold_check(accuracy,float(threshold_value["accuracy"]))

    return(str(deploy),)

# %%
from statistics import mode


@component(
    packages_to_install=["google-cloud-aiplatform", "sklearn"],
    base_image="python:3.9",
    output_component_file="deploy.yaml"
)
def deploy_model(
    model : Input[Model],
    project: str,
    region: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    
    ENDPOINT_NAME = "cancerModel"
    
    #Uploading model..
    deployed_model = aiplatform.Model.upload(
        display_name="XGBoost_Model",
        artifact_uri = model.uri.replace("model",""),
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest",
        serving_container_health_route=f"/v1/models/cancer",
        description = "1st version - Cancer",
        version_description="Cancer xgboost model",
    )

    def create_endpoint():
        endpoints = aiplatform.Endpoint.list(
        filter='display_name="{}"'.format(ENDPOINT_NAME),
        order_by='create_time desc',
        project=project, 
        location=region,
        )
        if len(endpoints) > 0:
            endpoint = endpoints[0]  # most recently created
        else:
            endpoint = aiplatform.Endpoint.create(
            display_name=ENDPOINT_NAME, project=project, location=region
        )
        return endpoint

    main_endpoint = create_endpoint()

    deployed_model_id = main_endpoint.list_models()
    
    if deployed_model_id is not None:
        for deployed_model in deployed_model_id:
            main_endpoint.undeploy(deployed_model_id=deployed_model.id)


    endpoint_deploy = deployed_model.deploy(machine_type="n1-standard-2", endpoint=main_endpoint, min_replica_count=1, max_replica_count=2, deployed_model_display_name="Cancer model one")
    # endpoint_deploy.wait()

    vertex_endpoint.uri = main_endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name

    deployed_model = aiplatform.Model.upload(
        display_name="XGBoost_Model",
        artifact_uri = model.uri.replace("model",""),
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest",
        serving_container_health_route=f"/v1/models/cancer",
        description = "2nd version - Cancer",
        version_description="Cancer xgboost model",
    )

    print("Model ID.............")
    print(main_endpoint.list_models()[0].id)

    deployed_model_id = main_endpoint.list_models()[0].id

    endpoint_deploy = deployed_model.deploy(machine_type="n1-standard-2", endpoint=main_endpoint, min_replica_count=1, max_replica_count=1, deployed_model_display_name="Cancer model two", traffic_split={"0":70,deployed_model_id:30})
    # endpoint_deploy.wait()

    


# %%
@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="getting-data",
    description="Getting data of breast cancer",
)
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION
):
    pulled_data = get_data().set_cpu_limit("1").set_memory_limit("2")
    train_eval = training(train_data=pulled_data.outputs["train_dataset"], test_data=pulled_data.outputs["test_dataset"], threshold_value_str={"accuracy":0.90}).set_cpu_limit("1").set_memory_limit("2").after(pulled_data)
    
    with dsl.Condition(
        train_eval.outputs["accuracy"] == "true",
        name="Testing Threshold"
    ):
        upload_model = deploy_model(model=train_eval.outputs["model"],project=project, region=region).set_cpu_limit("1").set_memory_limit("2")

if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=pipeline, package_path='xgb_pipe.json')
