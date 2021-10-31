# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().system('gcloud auth login')
get_ipython().system('gcloud projects list')


# %%
get_ipython().system('gsutil iam ch allUsers:objectViewer gs://vertextesting')


# %%
get_ipython().system('gsutil ls gs://vertextesting')


# %%
get_ipython().system('gcloud config set project vertextesting-330105')


# %%
PROJECT_ID = "vertextesting-330105"
REGION = "us-central1" #though us-central is cheaper
PIPELINE_ROOT = "gs://vertextesting/pipeline_root"


# %%
import kfp
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


# %%
@component(
    packages_to_install = [
        "pandas",
        "sklearn"
    ],
    base_image="python:3.9"
)
def get_data(
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset]
    
):
    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split as tts
    import pandas as pd
    # import some data to play with
    
    # data_raw = datasets.load_breast_cancer()
    data_raw = datasets.load_iris()
    data = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
    data["target"] = data_raw.target
    
    train, test = tts(data, test_size=0.3)
    train.to_csv(dataset_train.path)
    test.to_csv(dataset_test.path)


# %%
@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "xgboost",
        "joblib"
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
    
    data = pd.read_csv(dataset.path)
    print(data.columns)
    #data_numpy = data.to_numpy()
    #print("Numpy Data")
    #print(data_numpy)

    xgmodel = XGBClassifier(
        objective="multi:softmax",
        use_label_encoder=False
    )

    #target_val = data.target.astype('category')
    
    xgmodel.fit(
        data.drop(columns=["target","Unnamed: 0"]).values,
        data.target.values,
    )

    # xgmodel.fit(
    #     data.drop(columns=["target"]),
    #     data.target,
    # )

    # xgmodel.fit(
    #    data_numpy[:,:4],data_numpy[:,4]
    # )
    
    # score = xgmodel.score(
    #     data.drop(columns=["target"]),
    #     data.target,
    # )

    # score = xgmodel.score(
    #     data_numpy[:,:4],data_numpy[:,4]
    # )

    # model_artifact.metadata["train_score"] = float(score)
    # model_artifact.metadata["framework"] = "XGBoost"
    #metrics.log_metric("train_score",float(score))
    #dump(xgmodel, model.path + ".joblib")

    xgmodel.save_model(model.path + ".bst")


# %%
@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "xgboost"
    ],
    base_image="python:3.9"
)
def eval_model(
    test_set: Input[Dataset],
    xgb_model: Input[Model],
    metrics: Output[ClassificationMetrics],
    smetrics: Output[Metrics],
    final_model: Output[Model]
):
    from xgboost import XGBClassifier
    import pandas as pd
    
    data = pd.read_csv(test_set.path)
    model = XGBClassifier()
    model.load_model(xgb_model.path)
    
    score = model.score(
        data.drop(columns=["target"]),
        data.target,
    )
    
    from sklearn.metrics import roc_curve
    y_scores =  model.predict_proba(data.drop(columns=["target"]))[:, 1]
    fpr, tpr, thresholds = roc_curve(
         y_true=data.target.to_numpy(), y_score=y_scores, pos_label=True
    )
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())
    
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(data.drop(columns=["target"]))
    
    metrics.log_confusion_matrix(
       ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
       confusion_matrix(
           data.target, y_pred
       ).tolist(),  # .tolist() to convert np array to list.
    )
    
    xgb_model.metadata["test_score"] = float(score)
    smetrics.log_metric("score", float(score))

    final_model.metadata["Model"] = "v1"
    
    #model.save_model("gs://vertextesting/pipeline_root/model")


# %%
@component(
    packages_to_install=["google-cloud-aiplatform", "joblib", "sklearn"],
    base_image="python:3.9"
)
def deploy_model(
    model: Input[Model],
    project: str,
    region: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model]
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    a=model.uri.replace("model", ""),
    b=model_uri_address = ''.join(a)
    print("Model_path"+b)
    #model.uri.replace("model", "model.joblib")
    deployed_model = aiplatform.Model.upload(
        display_name="model-pipeline",
        #artifact_uri = b,
        artifact_uri = model.uri.replace("model", ""),
        #serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest"
        
    )

    main_endpoint= aiplatform.Endpoint.create(display_name="him",)
    
    endpoint = deployed_model.deploy(machine_type="n1-standard-2",endpoint=main_endpoint)

    endpoint.wait()
    
    ##Getting model endpoint id
    deployed_model_id = endpoint.list_models()[0].id

    deployed_model_two = aiplatform.Model.upload(
        display_name="model-pipeline-two",
        #artifact_uri = b,
        artifact_uri = model.uri.replace("model", ""),
        #serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest"
    )
    endpoint_two = deployed_model_two.deploy(machine_type="n1-standard-2",endpoint=main_endpoint,traffic_split={"0":60,deployed_model_id:40})
    # Save data to the output params
    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name


# %%
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
    train_op = train_xgb(dataset_op.outputs["dataset_train"])

    deploy_task = deploy_model(
        model=train_op.outputs["model"],
        project=project,
        region=region
    )

    # eval_op = eval_model(
    #     test_set=dataset_op.outputs["dataset_test"],
    #     xgb_model=train_op.outputs["model"]
    # ).after(deploy_model)

    # deploy_op = gcc_aip.ModelDeployOp( 
    #         # model=eval_op.outputs['final_model'],
    #         model = train_op.model,
    #         project=PROJECT_ID,
    #         machine_type="n1-standard-4",
    #     )

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='xgb_pipe.json')


# %%
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")


# %%
run1 = pipeline_jobs.PipelineJob(
    display_name="xgb_pipeline",
    template_path="xgb_pipe.json",
    job_id="xgb-pipeline-{0}".format(TIMESTAMP),
    # parameter_values={"bq_tab*le": "sara-vertex-demos.beans_demo.small_dataset"},
    enable_caching=True,
)


# %%
run1.run()


# %%



# %%
#project id and then endpoint id
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/880212553809/locations/us-central1/endpoints/3433159087032893440"
)


# %%
a=[6.2, 2.9, 4.3,1.3]


# %%
response = endpoint.predict([a])
print('Predicted: ', response.predictions)


# %%
deployed_model_id = endpoint.list_models()


# %%
deployed_model_id
#endpoint.undeploy(deployed_model_id=deployed_model_id)


# %%
# aiplatform.Model.resource_name


# %%



# %%
# from kfp.v2.google.client import AIPlatformClient

# api_client = AIPlatformClient(
#                 project_id=PROJECT_ID,
#                 region=REGION
#                 )

# response = api_client.create_run_from_job_spec(
#     'xgb_pipe.json',
#     enable_caching= False
# )


# %%



