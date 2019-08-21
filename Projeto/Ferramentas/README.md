# Sharelatex
Local Sharelatex tool using docker

## RUN
docker-compose up

## CREATE USER
docker exec sharelatex /bin/bash -c "cd /var/www/sharelatex; grunt user:create-admin --email joaopandolfi@gmail.com"
