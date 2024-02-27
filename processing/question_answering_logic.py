import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Define the model name for the pre-trained BERT model specialized in question
# answering tasks.
MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Initialize the tokenizer with the specified BERT model.
# The tokenizer is responsible for converting input text into a format that is
# compatible with the model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the pre-trained BERT model specified by MODEL_NAME.
# This model is fine-tuned on the SQuAD (Stanford Question Answering Dataset)
# for answering questions.
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)


def get_confidence(answer_start_scores, answer_end_scores):
    """
    Calculates the confidence score of the answer produced by
    the BERT model by analyzing the start and end logits.
    This function converts logits to probabilities using the
    softmax function and determines the confidence score
    based on the highest probabilities for the answer's
    start and end positions. The confidence score can be used
    to gauge the reliability of the answer, with higher
    scores indicating greater confidence in the model's response.

    Parameters:

    - answer_start_scores (torch.Tensor): The logits representing
      the model's predictions for the start position of the answer.
    - answer_end_scores (torch.Tensor): The logits representing the model's
      predictions for the end position of the answer.

    Returns:

    - float: The calculated confidence score for the model's answer,
      averaged between the start and end positions.

    Example:

    >>> get_confidence(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.1,
        0.2, 0.3]))
        0.3
    """
    # Convert the logits for answer start and end positions to probabilities
    # using softmax.
    start_probabilities = softmax(answer_start_scores, dim=-1)
    end_probabilities = softmax(answer_end_scores, dim=-1)

    # Determine the maximum probability for both start and end positions as an
    # indicator of confidence.
    max_start_prob = torch.max(start_probabilities)
    max_end_prob = torch.max(end_probabilities)

    # Calculate the confidence score by averaging the max probabilities of
    # start and end positions.
    confidence_score = (max_start_prob + max_end_prob) / 2

    # Return the confidence score as a float value.
    return confidence_score.item()


def answer_question_with_scibert(question, context):
    """
    Generates an answer to a given question based on a
    provided context using the SciBERT model.
    SciBERT, a BERT model trained on scientific texts,
    is utilized here to understand and extract
    answers from context that is likely dense with
    scientific terminology. This function encapsulates
    the process of encoding the question and context,
    performing inference with SciBERT, and decoding
    the model's prediction into a human-readable answer,
    along with computing a confidence score for
    the generated answer.

    Parameters:

    - question (str): The question to be answered.
    - context (str): The context or passage from which the answer should be
      derived.

    Returns:

    - tuple: A tuple containing the extracted answer as a string and its
      confidence score as a float.
      Returns a default response with a confidence of 0 if no context is
      provided.

    Example:
    
    >>> answer_question_with_scibert("What causes climate change?",
        "Climate change is caused by...")
        ("Climate change is caused by...", 0.95)
    """
    # Check if context is provided; if not, return a default message and zero
    # confidence.
    if not context:
        return ("Sorry, I couldn't find relevant"
                "information to answer your question."), 0

    # Define the maximum sequence length for SciBERT input.
    max_length = 512

    # Encode the question and context into tensor
    # inputs for SciBERT. Special tokens are added to mark
    # the beginning and end of the sequence, and the
    # input is truncated to the maximum length if necessary.
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True,
                                   max_length=max_length, truncation=True,
                                   return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # Perform inference with SciBERT without updating the model weights.
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the logits for the start and end positions of the
        # answer within the context.
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    # Identify the positions in the context that correspond to the start and
    # end of the answer.
    answer_start = torch.argmax(answer_start_scores)
    # Add 1 to include the end token in the range.
    answer_end = torch.argmax(answer_end_scores) + 1

    # Decode the identified answer span back into text using the tokenizer.
    answer = tokenizer.decode(input_ids[answer_start:answer_end])

    # Calculate the confidence score for the identified answer span.
    confidence = get_confidence(answer_start_scores, answer_end_scores)

    # Return the decoded answer text, stripped of any leading/trailing
    # whitespace, along with the confidence score.
    return answer.strip(), confidence
