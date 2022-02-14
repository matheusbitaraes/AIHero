#export DOCKER_HOST=..
docker build --tag ai-hero:v1 .
eval $(minikube docker-env)
minikube cache add ai-hero:v1
helm upgrade ai-hero-chart ai-hero-chart/.