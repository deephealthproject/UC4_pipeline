sudo docker build  -t riccardorenzulli/deephealth-uc4:pytorch . -f Dockerfile_pytorch
sudo docker push riccardorenzulli/deephealth-uc4:pytorch
#kubectl apply -f pytorch/pod-pytorch-deephealth-uc4-training-4gpu-noblackmasks.yaml
