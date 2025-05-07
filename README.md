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
***************************************kubeflow**************************************************************
Resources: Ensure your EC2 instance has at least 4 CPUs, 16GB RAM, and 50GB disk
(e.g., t3.xlarge or larger). Check with:
**************************************************************************************************************
lscpu
free -m
df -h
 Install Kustomize (if not already)

curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

git clone https://github.com/kubeflow/manifests.git
cd manifests
git checkout v1.7-branch


while ! kustomize build example | kubectl apply -f -; do echo "Retrying..."; sleep 10; done


kubectl get pods -n kubeflow

Port-Forward Kubeflow Central Dashboard

kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 --address 0.0.0.0

Now open your browser to:

http://Ec2-publicip:8080/
u:  user@example.com
p: 12341234
************************************************************************************************************
minikube stop
minikube delete
minikube start --cpus=4 --memory=8192 --disk-size=50g --driver=docker  <------------------------------

kubectl get nodes
kubectl cluster-info
****************************************************************************************************************

