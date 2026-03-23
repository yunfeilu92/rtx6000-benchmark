#!/bin/bash
set -e

# === G7E.48xlarge Instance Launch Script ===
# GPU: 8x NVIDIA RTX PRO Server 6000 (96GB GDDR7 each, Blackwell SM 12.0)
# Price: ~$33.14/hr On-Demand
#
# Availability (as of 2026-03):
#   us-east-1: us-east-1b, us-east-1d (often out of capacity)
#   us-east-2: us-east-2a, us-east-2b
#   us-west-2: us-west-2a, us-west-2b, us-west-2c, us-west-2d (most AZs)

REGION="${1:-us-west-2}"
INSTANCE_TYPE="g7e.48xlarge"

# Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
# Find latest: aws ec2 describe-images --region $REGION --owners amazon \
#   --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
#   --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name]' --output text
AMI_ID=""
KEY_NAME=""
SUBNET_ID=""  # Leave empty to let AWS pick AZ
SG_ID=""

# Auto-detect AMI if not set
if [ -z "$AMI_ID" ]; then
    echo "Auto-detecting latest DLAMI in $REGION..."
    AMI_ID=$(aws ec2 describe-images --region $REGION --owners amazon \
        --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)
    echo "AMI: $AMI_ID"
fi

if [ -z "$KEY_NAME" ]; then
    echo "ERROR: Please set KEY_NAME before launching."
    echo "  List keys: aws ec2 describe-key-pairs --region $REGION --query 'KeyPairs[*].KeyName' --output text"
    exit 1
fi

echo "=== Launching $INSTANCE_TYPE in $REGION ==="

aws ec2 run-instances \
    --region $REGION \
    --instance-type $INSTANCE_TYPE \
    --image-id $AMI_ID \
    --key-name $KEY_NAME \
    ${SUBNET_ID:+--subnet-id $SUBNET_ID} \
    ${SG_ID:+--security-group-ids $SG_ID} \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=g7e-benchmark},{Key=Project,Value=rtx6000-benchmark}]" \
    --output json | tee /tmp/g7e-launch.json

INSTANCE_ID=$(cat /tmp/g7e-launch.json | python3 -c "import sys,json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")
echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "=== Instance Ready ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "SSH: ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Next steps:"
echo "  1. scp setup.sh download_data.sh run_benchmark.sh ubuntu@${PUBLIC_IP}:~/"
echo "  2. ssh ubuntu@${PUBLIC_IP}"
echo "  3. bash setup.sh"
echo "  4. bash download_data.sh"
echo "  5. bash run_benchmark.sh"
