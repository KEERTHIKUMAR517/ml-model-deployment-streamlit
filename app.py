import streamlit as st
import os
import boto3
from transformers import pipeline
import torch

# Load from Streamlit secrets
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

bucket_name = 'mlops-bucket-2025'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

local_path = 'tiny-bert-streamlit'
s3_prefix = 'ml-models/tinybert-sentiment-analysis'

# Use secrets
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
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                s3.download_file(bucket_name, s3_key, local_file)

st.title('ML model Deployment')
button = st.button('Download model')

if button:
    with st.spinner('Downloading... PLzz wait'):
        download_dir(local_path, s3_prefix)
        st.success('Model Downloaded ✅')

text = st.text_area('Enter your review', 'Type here')

predict = st.button('Predict')
classifier = pipeline('text-classification', model='tiny-bert-streamlit', device=device)

if predict:
    with st.spinner('Predicting output...'):
        output = classifier(text)
        st.write(output)
