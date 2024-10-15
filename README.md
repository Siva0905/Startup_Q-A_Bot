**Startup Advisor Chatbot**

This project is a chatbot built using Flask that provides answers to questions related to startups. It leverages Sentence-BERT to find similar questions from a dataset and DistilBERT to generate answers. The chatbot provides a web interface where users can input questions and receive the most relevant responses.

Features
Sentence-BERT: Retrieves the most similar question from a pre-defined dataset using cosine similarity between question embeddings.
DistilBERT: Generates answers based on the context of the most similar question.
Flask Web Interface: User-friendly interface for interacting with the chatbot.

Installation
Prerequisites

Python 3.8+
Flask
PyTorch
Transformers library
Sentence-Transformers library

Installation Steps:

Clone the repository and navigate to the project directory.
Install the required dependencies listed in requirements.txt.
Download or place the fine-tuned DistilBERT model in the appropriate directory (./results).
Ensure your dataset is in JSON format and available at the specified path (or update the path in the code).
Dataset
The dataset consists of questions, their corresponding answers, and the context in which the answers are found. The chatbot uses this dataset to find similar questions and generate answers based on the context. 

The dataset must follow a structured JSON format containing the following fields:

context: A paragraph or article providing relevant information.
questions: A set of questions related to the context.
answers: The correct answers corresponding to the questions.
Application Overview
This chatbot uses two key models:

Sentence-BERT: This is used to match the user's question with the most similar one from the dataset by computing cosine similarity between question embeddings.
DistilBERT: After retrieving the most similar question, this model extracts an answer based on the context associated with that question.
Workflow
The user enters a question via the web interface.
Sentence-BERT finds the most similar question from the dataset.
DistilBERT generates an answer using the relevant context.
The chatbot displays the similar question, context, predicted answer, and the correct answer from the dataset.
Usage
Running the Application
Start the Flask application, and access the chatbot by navigating to http://127.0.0.1:5000/ in a web browser.
Input your question into the text box and submit to receive a response.
User Interface
The chatbot interface includes a text field for question input and a submit button. The response includes:
The most similar question found from the dataset.
The context in which the answer can be found.
The generated answer based on the context.
The correct answer from the dataset for comparison.
API Endpoints
/
Provides the web interface for interacting with the chatbot.
/answer (POST)
Accepts a question in JSON format, finds the most similar question, and generates an answer.
The response includes the similar question, context, generated answer, and correct answer.
Contributing
Contributions to this project are welcome. If you'd like to add new features or fix any issues, follow these steps:

Fork the repository.
Create a feature branch.
Commit your changes.
Push your branch and create a pull request.
