# Kubeflow-pipeline
Kubeflow-pipeline


Minimum Requirements (for development/single user testing)

Component	              vCPUs	    RAM	         Disk

Kubeflow (full)	         4	       8–16 GB	       50+ GB

Minikube install	        2	       8 GB	          30+ GB


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

Edit Security Limits

sudo nano /etc/security/limits.conf

Add at the bottom:

* soft nofile 65536
* hard nofile 65536

🔹 Step 2: Enable PAM limits support

Edit this file:



sudo nano /etc/pam.d/common-session

Add this line if not already present:

session required pam_limits.so

Do the same for:

sudo nano /etc/pam.d/common-session-noninteractive

🔹 Step 3: Update systemd configuration

Create or edit this file:

sudo mkdir -p /etc/systemd/system.conf.d

sudo nano /etc/systemd/system.conf.d/limits.conf

Add:

[Manager]

DefaultLimitNOFILE=65536

Also, for the kubelet service:


sudo mkdir -p /etc/systemd/system/kubelet.service.d

sudo nano /etc/systemd/system/kubelet.service.d/limits.conf
Add:


[Service]

LimitNOFILE=65536

🔹 Step 4: Reload and restart systemd services

sudo systemctl daemon-reexec

sudo systemctl daemon-reload

sudo systemctl restart kubelet

🔹 Step 5: Reboot the instance

This step is essential to apply all the ulimit changes correctly:


sudo reboot

🔹 Step 6: Verify the new file limit after reboot

Once back online, SSH into your instance and run:

ulimit -n

You should see:

65536

kubectl delete pod -n kubeflow -l app=admission-webhook


************************************************************************************************************

