
import logging
import os
import sys
import threading
from transformers import logging as hf_logging

# Surpress additional messages from appearing in the terminal when talking
# to the chatbot, comment the two lines below out for debugging.
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.ERROR)

from interaction_messages.interaction_messages import (check_spelling,
                                                       humanized_response,
                                                       introduction,
                                                       show_processing)
from utils.utils import format_question, validate_question
from processing.context_retrival import search_and_evaluate_answers
from processing.keyword_extraction import extract_and_process_keywords





def generate_response(question):
    """
    Processes a given question by extracting relevant keywords,
    searching for answers across multiple data sources,
    and evaluating the confidence of these answers. It leverages
    a combination of keyword extraction, search, and
    answer evaluation techniques to attempt to provide the most
    accurate and relevant answer possible. If a confident
    answer is found, it is returned; otherwise, a prompt for
    further clarification is provided.

    Parameters:

    - question (str): The question posed by the user.

    Returns:
    
    - str: An answer found for the question above a given threshold of
      confidence, or a message indicating that further clarification is needed.
    """
    # Log the processing of a new question to provide traceability and
    # debugging support.
    logging.info(f"Processing question: {question}")

    # Extract and process keywords from the question for targeted search
    # and evaluation.
    accurate_keywords = extract_and_process_keywords(question)

    # Search across available data sources using the extracted keywords
    # and evaluate the answers found for confidence.
    response, found_answer = search_and_evaluate_answers(
        question, accurate_keywords)

    # Check if a confident answer was found based on the evaluation criteria.
    if found_answer:
        # If a confident answer is found, return it directly.
        return response, True
    else:
        # If no confident answer is found, return a message prompting the
        # user for more details to improve the search.
        return ("After exploring various contexts, I couldn't find "
                "a precise answer. Could you provide more details "
                "or clarify your question?"), False



def start_chat_bot():
    """
    Initiates the chatbot for users to interact with.
    This function handles the lifecycle of a chat session,
    including greeting the user, processing user queries,
    and managing the chatbot's responses. It leverages
    multithreading to display a processing indicator while
    generating responses to user questions, enhancing the
    user experience by indicating that their query is being processed.

    The chatbot session continues until the user decides to exit by
    typing 'exit'. Each question posed by the user
    is validated, spell-checked, and then passed on for response
    generation, with the chatbot providing either a
    direct answer or an acknowledgment of inability to find an answer.
    """
    # Redirect standard error to a null device to suppress error messages.
    # Comment the below line out in order to debug.
    sys.stderr = open(os.devnull, 'w')
    # Display the chatbot's introduction message.
    introduction()

    # Enter a loop to continuously interact with the user.
    while True:

        # Ask for the users question.
        user_question = input("\nAsk your question (type 'exit' to quit): ")

        # Check if the user wishes to exit the chatbot session.
        if user_question.lower() == "exit":
            print("Thanks for stopping by. Goodbye!")
            break  # Exit the loop and end the session.

        # Validate the user's question to ensure it meets the criteria.
        if not validate_question(user_question):
            # Skip further processing and prompt for another question.
            continue

        # Asks the user to correct spelling errors in the question before
        # processing.
        # The user may continue without correcting the spelling if they feel
        # it is correct.
        corrected_question = check_spelling(user_question)
        
        # Remove leading or trailing white space and add a question mark as
        # this can help the BERT model.
        corrected_question = format_question(corrected_question)

        # Initialize a threading event to manage the processing indicator.
        stop_event = threading.Event()

        # Start a separate thread to show a processing indicator while
        # generating the response.
        processing_thread = threading.Thread(
            target=show_processing, args=(stop_event,))
        processing_thread.start()

        # Generate a response to the user's question.
        answer, found_answer = generate_response(corrected_question)

        # Signal the processing indicator thread to stop and wait for
        # it to finish.
        stop_event.set()
        processing_thread.join()

        # Print the chatbot's response to the user's question.
        print("\n", end="")
        if found_answer:
            # Format the answer in a more human-readable way by adding
            # additional text to the answer.
            response = humanized_response(answer)
            print(response)
        else:
            # If no answer was found, print the default response generated by
            # the chatbot.
            print(answer)


if __name__ == "__main__":
    start_chat_bot()
