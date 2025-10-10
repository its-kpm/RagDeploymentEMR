#!/bin/bash
set -e

# 1️⃣ Install Java 17
sudo yum install -y java-17-amazon-corretto

# 2️⃣ Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
echo "export PATH=$PATH:/usr/local/bin" >> ~/.bashrc
source ~/.bashrc

# 3️⃣ Pull embedding + chat models (do this here so they’re ready when API starts)
ollama pull mxbai-embed-large
ollama pull llama3

# 4️⃣ Download your JAR from the correct S3 bucket
aws s3 cp s3://myhadoopemr/RagDeployment-assembly-0.1.0.jar /home/hadoop/RagDeployment-assembly-0.1.0.jar