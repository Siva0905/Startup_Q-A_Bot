from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer, util
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import json

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = DistilBertForQuestionAnswering.from_pretrained('./results')
tokenizer = DistilBertTokenizer.from_pretrained('./results')

# Load the dataset
dataset_file = 'D:/Altruisty_Chatbot/augmented_startup_updated_qs.json'
with open(dataset_file, 'r') as f:
    data = json.load(f)

# Load Sentence-BERT model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Prepare question embeddings
question_embeddings = []
for article in data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            question_embeddings.append((qa['question'], paragraph['context'], qa['answers'][0]['text']))

question_texts = [q[0] for q in question_embeddings]
question_embeddings_sbert = sbert_model.encode(question_texts, convert_to_tensor=True)

# Function to find the most similar question
def find_similar_question(question):
    question_embedding = sbert_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, question_embeddings_sbert)[0]
    best_match_idx = torch.argmax(similarities).item()
    return question_embeddings[best_match_idx]

# Function to answer a question using the trained model
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    input_ids = inputs['input_ids'].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

@app.route('/')
def index():
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot Integration</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f4f4f4;
            }
            .container {
                max-width: 600px;
                margin: auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .question-input {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .submit-btn {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .submit-btn:hover {
                background-color: #45a049;
            }
            .response {
                margin-top: 20px;
                padding: 10px;
                background: #e0f7fa;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Startup Advisor Chatbot</h1>
            <input type="text" id="question" class="question-input" placeholder="Enter your question">
            <button class="submit-btn" onclick="askQuestion()">Submit</button>
            <div id="response" class="response"></div>
        </div>

        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                const responseDiv = document.getElementById('response');

                if (!question) {
                    responseDiv.innerHTML = 'Please enter a question.';
                    return;
                }

                const response = await fetch('/answer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: question })
                });

                if (response.ok) {
                    const data = await response.json();
                    responseDiv.innerHTML = `
                        <strong>Similar Question:</strong> ${data.similar_question}<br>
                        <strong>Context:</strong> ${data.context}<br>
                        <strong>Answer:</strong> ${data.answer}<br>
                        <strong>Correct Answer:</strong> ${data.correct_answer}
                    `;
                } else {
                    responseDiv.innerHTML = 'Error fetching the answer. Please try again.';
                }
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_content)

@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    similar_question, context, correct_answer = find_similar_question(question)
    answer = answer_question(question, context)

    response = {
        'similar_question': similar_question,
        'context': context,
        'question': question,
        'answer': answer,
        'correct_answer': correct_answer
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
