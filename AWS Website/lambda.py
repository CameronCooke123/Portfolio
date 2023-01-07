import json
import boto3
import random

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('F2LPractice')

caseList = list(range(0, table.item_count))
lastIndex = len(caseList)

def lambda_handler(event, context):
    global caseList
    global lastIndex
    
    # When the webpage is first loaded/refreshed, the case list should reset.
    if event['resetList'] == 'true':
        caseList = list(range(0, table.item_count))
        lastIndex = len(caseList)
    # We want each case to appear exactly once before any case repeats, but we
    # want them to appear in an otherwise random order. To do this, we take a 
    # list with integers from 0 to the number of cases, and shuffle them.
    if lastIndex >= len(caseList):  
        random.shuffle(caseList)
        lastIndex = 0
    case = caseList[lastIndex]
    response = table.get_item(Key={'caseNum': case})
    lastIndex += 1
    return {
        'caseNum': json.dumps(case),
        'scrambleAlg': json.dumps(response['Item']['scrambleAlg']),
        'solutionAlg': json.dumps(response['Item']['solutionAlg'])
    }
    
