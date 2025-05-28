import streamlit as st
import os
import boto3
from transformers import pipeline
import torch

# Load from Streamlit secrets
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

bucket_name = 'mlops-bucket-2025'
device = 0 if torch.cuda.is_available() else -1  # HuggingFace expects int, not torch.device

local_path = 'tiny-bert-streamlit'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

# Use secrets to create S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                if s3_key.endswith('/'):
                    continue  # Skip folders
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)

st.title('ML Model Deployment')
button = st.button('Download model')

if button:
    with st.spinner('Downloading... Please wait'):
        download_dir(local_path, s3_prefix)
        st.success('Model Downloaded ✅')

text = st.text_area('Enter your review', 'Type here')
predict = st.button('Predict')

if predict:
    with st.spinner('Predicting output...'):
        classifier = pipeline('text-classification', model=local_path, device=device)
        output = classifier(text)
        st.write(output)
