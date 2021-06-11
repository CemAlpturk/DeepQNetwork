FROM ubuntu:20.04
# FROM ubuntu:18.04
# FROM ubuntu

# Install python3 and pip3
# RUN apt-get update --fix-missing
# RUN apt-get install -y apt-transport-https
RUN apt-get -y update
RUN apt-get install -y python3 python3-pip

# Update pip, required for installing Tensorflow 2
RUN pip3 install --upgrade pip

# Copy over requirements to container
COPY src/requirements.txt env/requirements.txt

# Install requirements
RUN pip3 install -r env/requirements.txt

# Need to install gifsicle separately for Ubuntu
RUN apt-get install gifsicle -y
