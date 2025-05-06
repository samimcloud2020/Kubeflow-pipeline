# Kubeflow-pipeline
Kubeflow-pipeline

*************************************************************************************************
kubectl -n kubernetes-dashboard edit service kubernetes-dashboard
Change this section from:

type: ClusterIP
To:

type: NodePort

kubectl -n kubernetes-dashboard get svc kubernetes-dashboard   <-----got 31900
kubectl -n kubernetes-dashboard port-forward service/kubernetes-dashboard 31900:80 --address=0.0.0.0

*******************************************************************************************************
**************************kubectl install*************************************************************
curl -LO https://dl.k8s.io/release/v1.30.1/bin/linux/amd64/kubectl
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
kubectl version --client
*****************************************************************************************************
*****************Minikube install*******************************************************************
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube version
***********************************start minikube**************************************************
Option A: Using Docker
minikube start --driver=docker

Option B: Using VirtualBox
minikube start --driver=virtualbox


If you don’t have Docker:
sudo apt install docker.io -y
sudo usermod -aG docker $USER
newgrp docker
*****************************************************************************************************
kubectl get nodes
*****************************************************************************************************


