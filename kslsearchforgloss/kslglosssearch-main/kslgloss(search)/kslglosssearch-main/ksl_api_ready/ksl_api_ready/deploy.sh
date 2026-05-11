#!/bin/bash
# KSL Glosser - Google Cloud Run Deployment Script
# Author: Wambui
# Usage: ./deploy.sh [PROJECT_ID]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${1:-signvrse-project}"  # Use first argument or default
SERVICE_NAME="ksl-glosser"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  KSL Glosser - Google Cloud Run Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Project ID: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found${NC}"
    echo "Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Confirm deployment
read -p "Proceed with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

echo ""
echo -e "${YELLOW}Step 1/6: Setting GCP project...${NC}"
gcloud config set project ${PROJECT_ID}

echo ""
echo -e "${YELLOW}Step 2/6: Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com

echo ""
echo -e "${YELLOW}Step 3/6: Building container image...${NC}"
echo "This may take 5-10 minutes..."
gcloud builds submit --tag ${IMAGE_NAME} --timeout=20m

echo ""
echo -e "${YELLOW}Step 4/6: Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --min-instances 0 \
    --timeout 60s \
    --concurrency 80 \
    --allow-unauthenticated \
    --set-env-vars ENVIRONMENT=production

echo ""
echo -e "${YELLOW}Step 5/6: Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo -e "${YELLOW}Step 6/6: Testing deployment...${NC}"
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s ${SERVICE_URL}/health)

if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo -e "${GREEN}Health check passed!${NC}"
else
    echo -e "${RED}Health check failed${NC}"
    echo "Response: ${HEALTH_RESPONSE}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo "API Docs: ${SERVICE_URL}/docs"
echo "Health Check: ${SERVICE_URL}/health"
echo ""
echo "Test the API:"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo "  curl -X POST ${SERVICE_URL}/search \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"Where is the bathroom?\", \"top_k\": 3}'"
echo ""
echo "View logs:"
echo "  gcloud logs read --service=${SERVICE_NAME} --limit=50"
echo ""
echo "Monitor service:"
echo "  https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}"
echo ""