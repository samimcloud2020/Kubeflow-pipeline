from typing import Dict, List
from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, component


@dsl.component(base_image="python:3.9")
def load_data(output_csv: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    from sklearn.datasets import load_iris
    import pandas as pd
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv(output_csv.path, index=False)


@dsl.component(base_image="python:3.9")
def preprocess_data(input_csv: Input[Dataset], output_train: Output[Dataset], output_test: Output[Dataset],
                    output_ytrain: Output[Dataset], output_ytest: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(input_csv.path)
    df = df.dropna()

    X = df.drop(columns=['target'])
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    pd.DataFrame(X_train, columns=X.columns).to_csv(output_train.path, index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv(output_test.path, index=False)
    pd.DataFrame(y_train).to_csv(output_ytrain.path, index=False)
    pd.DataFrame(y_test).to_csv(output_ytest.path, index=False)


@dsl.component(
    base_image="python:3.9",
    output_component_file="train_model_component.yaml",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model_and_save_to_pvc(train_data: Input[Dataset], ytrain_data: Input[Dataset]):

    import os
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from joblib import dump

    X_train = pd.read_csv(train_data.path)
    y_train = pd.read_csv(ytrain_data.path)

    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())

    pvc_path = "/mnt/models/iris/1"
    os.makedirs(pvc_path, exist_ok=True)
    dump(model, os.path.join(pvc_path, "model.joblib"))


@dsl.component(
    base_image="google/cloud-sdk:slim",
)
def deploy_kserve():
    kserve_yaml = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: iris-model
spec:
  predictor:
    sklearn:
      storageUri: pvc://model-pvc/iris
      resources:
        requests:
          cpu: 100m
          memory: 256Mi
"""
    with open("inference.yaml", "w") as f:
        f.write(kserve_yaml)

    import subprocess
    subprocess.run(["kubectl", "apply", "-f", "inference.yaml"], check=True)


@dsl.pipeline(name="ml-pipeline-with-kserve-pvc")
def ml_pipeline():
    load_op = load_data()
    preprocess_op = preprocess_data(
        input_csv=load_op.outputs["output_csv"]
    )
    train_op = train_model_and_save_to_pvc(
        train_data=preprocess_op.outputs["output_train"],
        ytrain_data=preprocess_op.outputs["output_ytrain"]
    ).set_volume_mounts([dsl.VolumeMount(mount_path="/mnt/models", name="model-pvc")])

    deploy_op = deploy_kserve().after(train_op)


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path="kubeflow_pipeline_with_pvc.yaml")
