#export DOCKER_HOST=..
docker build --tag ai-hero-api .
kubectl apply -f api-deploy.yml
kubectl rollout restart deployment/ai-hero-api