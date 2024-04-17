import boto3
import json, os

ENDPOINT = os.environ['ENDPOINT']
runtime = boto3.client('runtime.sagemaker')

def respond(err, res=None):
    return {
        'statusCode': '400' if err else '200',
        'body': str(err) if err else json.dumps(res),
        'headers': {
            'Content-Type': 'application/json',
        },
    }

def lambda_handler(event, context):
    
    # data = (json.loads(event["body"]))["body"]
    data = event['body']
    
    payload = json.dumps({
        "body": data
    })
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='application/json', Body=payload)
    result = response['Body'].read().decode('utf-8')
    
    payload = {
        "cartoonized_img": result   
    }
    
    return respond(None, payload)


