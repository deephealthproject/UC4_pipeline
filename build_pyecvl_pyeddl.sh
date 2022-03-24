sudo docker build  -t riccardorenzulli/deephealth-uc4:pyecvl-pyeddl . -f Dockerfile_pyecvl_pyeddl
sudo docker push riccardorenzulli/deephealth-uc4:pyecvl-pyeddl
#kubectl apply -f eddl/pod-deephealth-uc4-training-4gpu.yaml
