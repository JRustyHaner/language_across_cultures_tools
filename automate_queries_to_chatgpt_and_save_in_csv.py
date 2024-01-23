#this script uses the chatgpt api to generate responses to a list of queries combined with a list of contexts and saves the responses in a csv file

#import libraries
import requests
import json
import csv
import time
import openai

#there is usage, if no arguments are passed
# it should be automate_queries_to_chatgpt_and_save_in_csv.py <queries_csv_file> <contexts_csv_file> <output_csv_file>
# or if no api key is found in key.txt file, it should be automate_queries_to_chatgpt_and_save_in_csv.py <queries_csv_file> <contexts_csv_file> <output_csv_file> <api_key>

#read the arguments
def read_arguments():
    import sys
    if len(sys.argv) == 4:
        queries_csv_file = sys.argv[1]
        contexts_csv_file = sys.argv[2]
        output_csv_file = sys.argv[3]
        api_key = ""
    elif len(sys.argv) == 5:
        queries_csv_file = sys.argv[1]
        contexts_csv_file = sys.argv[2]
        output_csv_file = sys.argv[3]
        api_key = sys.argv[4]
    else:
        print("Usage: automate_queries_to_chatgpt_and_save_in_csv.py <queries_csv_file> <contexts_csv_file> <output_csv_file> <api_key>")
        exit()
    return queries_csv_file, contexts_csv_file, output_csv_file, api_key

#read the api key from the key.txt file
def read_api_key(api_key):
    if api_key == "":
        try:
            with open("key.txt", "r") as f:
                api_key = f.read()
        except:
            print("No api key found in key.txt file")
            exit()
    return api_key

#get the api key from the arguments and save it to the key.txt file
def define_api_key(api_key):
    if api_key != "":
        with open("key.txt", "w") as f:
            f.write(api_key)
    return api_key

#read the queries from the csv file, its just a comma separated list of queries
def read_queries(queries_csv_file):
    queries = []
    with open(queries_csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            queries.append(row[0])
    return queries

#read the contexts from the csv file, its just a comma separated list of contexts
def read_contexts(contexts_csv_file):
    contexts = []
    with open(contexts_csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            contexts.append(row[0])
    return contexts

#combine the queries and contexts into a list of queries with contexts
def combine_queries_and_contexts(queries, contexts):
    queries_with_contexts = []
    for query in queries:
        for context in contexts:
            queries_with_contexts.append(query + " " + context)
    return queries_with_contexts

#generate the responses from the chatgpt api
def generate_responses_from_chatgpt_api(queries_with_contexts, api_key):
    responses = []
    for query_with_context in queries_with_contexts:
        response = generate_response_from_chatgpt_api(query_with_context, api_key)
        responses.append(response)
        time.sleep(1)
    return responses

#generate the response from the chatgpt api
def generate_response_from_chatgpt_api(query_with_context, api_key):
    openai.api_key = api_key
    messages = [{"role": "user", "content": query_with_context}]
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message["content"]

#save the responses to the csv file
def save_responses_to_csv_file(queries_with_contexts, responses, output_csv_file):
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(queries_with_contexts)):
            writer.writerow([queries_with_contexts[i], responses[i]])

#main function
def main():
    queries_csv_file, contexts_csv_file, output_csv_file, api_key = read_arguments()
    #if the api key is not passed as an argument, read it from the key.txt file
    if api_key == "":
        api_key = read_api_key(api_key)
    #if the api key is passed as an argument, save it to the key.txt file
    else:
        api_key = define_api_key(api_key)
    queries = read_queries(queries_csv_file)
    contexts = read_contexts(contexts_csv_file)
    queries_with_contexts = combine_queries_and_contexts(queries, contexts)
    responses = generate_responses_from_chatgpt_api(queries_with_contexts, api_key)
    save_responses_to_csv_file(queries_with_contexts, responses, output_csv_file)

