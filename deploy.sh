#export DOCKER_HOST=..
docker build --tag ai-hero:v1 .
minikube cache add ai-hero:v1
helm upgrade ai-hero-chart ai-hero-chart/.