#!/bin/python

from flask import Flask, jsonify, request, Response
from ctransformers import AutoModelForCausalLM
import yaml
from flask import Flask, request, jsonify, abort, make_response
import queue
import threading
from time import time
import itertools
import os

app = Flask(__name__)

def get_config():
    with open('config.yml', 'r') as file:
        return yaml.safe_load(file)
    
def generate(llm, prompt):
    input_tokens = len(llm.tokenize(prompt))
    yield f"""# Generating response from [{input_tokens}] tokens: """
    response = ""
    start_time = time()
    for chunk in llm(prompt, stream=True):
        response += chunk
        yield chunk
    end_time = time()
    elapsed_time = end_time - start_time
    output_tokens = len(llm.tokenize(response))
    yield f"""\n# Generated [{output_tokens}] tokens in [{round(elapsed_time, 1)}s]\n"""

config = get_config()
model_path=os.environ.get('MODEL_PATH')
model = None
llm = None
task_queue = queue.Queue()
results = {}
condition = threading.Condition()

def worker():
    while True:
        with condition:
            while task_queue.empty():
                condition.wait()
            job = task_queue.get()

        client_addr, queue_model, prompt = job
        global model, llm
        if model != queue_model:
            model = queue_model
            del llm
            llm = None
        if llm is None:
            del llm
            llm = AutoModelForCausalLM.from_pretrained(model_path + model['location'], 
                                               max_new_tokens=model['max_new_tokens'],
                                               model_type=model['model_type'],
                                               gpu_layers=model['gpu_layers'])
        result = generate(llm, prompt)

        with condition:
            results[client_addr] = result
            condition.notify_all()

worker_thread = threading.Thread(target=worker)
worker_thread.start()

@app.route("/")
def index():
    return "REST API"

@app.route("/config")
def getConfig():
    return jsonify(config)

@app.route('/answer')
def answer():
    query = request.args.get('query')
    model_num = int(request.args.get('model_num', 1))
    model = config['models'][model_num]
    client_addr = request.remote_addr
    prompt = model['prompts'][0].replace("{query}", query)

    with condition:
        task_queue.put((client_addr, model, prompt))
        condition.notify_all()

    # Block until this client's result is ready
    with condition:
        while client_addr not in results:
            condition.wait()

    result = results.pop(client_addr)

    return Response(itertools.chain(["# Prompt: "+query + "\n"], result), mimetype='text/plain')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
