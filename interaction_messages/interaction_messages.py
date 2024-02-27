import random
import time
import re
from spellchecker import SpellChecker


def introduction():
    """
    Displays a welcoming message to the user
    upon starting the chatbot session.
    This function sets the tone for the user
    interaction, encouraging users to ask
    questions related to science. It guides
    users on how to phrase their questions
    by suggesting starting keywords, aiming
    to foster an engaging and informative
    dialogue about various scientific topics.

    The introduction is designed to be friendly
    and informative, setting expectations
    for the types of queries the chatbot is
    equipped to handle and inviting users to
    explore scientific knowledge together.
    """
    # Print the welcoming and instructional message to the user.
    print("Welcome to our Science Chatbot! 🌍✨ "
          "Dive into the wonders of science by asking me questions. "
          "I'm here to unearth answers and insights for you. "
          "Just start your question with keywords like 'What', 'Why', 'When', "
          "'Where', 'Who', 'How', 'Explain', or 'Define' to get started. "
          "Let's explore the mysteries of the universe together! 🚀🔭")


def show_processing(stop_event):
    """
    Displays a dynamic processing indicator to the user
    while the chatbot is generating a response.
    This function aims to enhance the user experience by
    providing visual feedback that their query
    is being processed. The indicator consists of a
    simple text animation that simulates a loading
    effect with dots.

    Parameters:

    - stop_event (threading.Event): An event that signals
      when to stop the animation, indicating
      that processing has completed.

    The function runs in a separate thread and
    continuously updates the processing message until
    the stop_event is set, at which point it
    cleans up the message before terminating.
    """
    # Define a sequence of dot patterns for the loading animation.
    dots = ["", ".", "..", "..."]
    idx = 0

    # Continue updating the processing message until signaled to stop.
    while not stop_event.is_set():
        # Clear the line before displaying the new processing message.
        print("\r" + " " * 20, end="")

        # Now, print the new processing message starting fresh on the same
        # line.
        print("\rProcessing" + dots[idx % len(dots)], end="")
        idx += 1

        # Pause for a second before the next update to create the animation
        # effect.
        time.sleep(1)

    # Once processing is complete, clear the line to tidy up the console
    # output.
    print("\r" + " " * 20, end="")


def check_spelling(question):
    """
    Checks the spelling of words within a user's question
    and offers an opportunity for correction.
    This function aims to improve the chatbot's
    understanding and response accuracy by ensuring questions
    are free of spelling errors. It uses the SpellChecker
    library to identify misspelled words and prompts
    the user to correct them, enhancing the clarity and
    precision of queries processed by the chatbot.
    We could autocorrect the words, however, given the
    questions are meant to be scientific some terms
    may appear to be spelt incorrectly when they are in fact not.

    Parameters:

    - question (str): The user's original question.

    Returns:

    - str: The user-corrected question if corrections were made, or the
      original question otherwise.

    Example:

    >>> check_spelling("What is photosyntheis?")
        "There seems to be a spelling error in your question: photosyntheis"
        "Please resend the correct message or press enter to continue anyway: "
        "What is photosynthesis?"
    """
    # Initialize the SpellChecker.
    spell = SpellChecker()

    # Extract words from the question, ignoring punctuation and converting to
    # lowercase for consistency.
    words = re.findall(r'\b\w+\b', question.lower())

    # Identify words not recognized by the SpellChecker.
    misspelled = spell.unknown(words)

    # If there are misspelled words:
    if misspelled:
        # Compile a list of the misspelled words to inform the user.
        misspelled_words = ', '.join(word for word in misspelled)
        print(
            f"There seems to be a spelling error in your"
            f" question: {misspelled_words}")

        # Prompt the user for a corrected question or to continue with the
        # original question.
        correction = input(
            "Please resend the correct message "
            "or press enter to continue anyway: "
        )

        # Return the corrected question if provided, or the original question
        # if not.
        return correction if correction else question
    else:
        # Return the original question if no spelling errors were found.
        return question


def humanized_response(answer):
    """
    Generates a humanized response to the user's
    question with the provided answer.
    This function selects a response template from a
    predefined list, incorporating the answer into
    a more conversational and engaging format. The
    aim is to make the chatbot's responses feel more
    natural and less robotic, enhancing the overall
    user experience by adding variety and a touch of
    personality to the chatbot's communication style.

    Parameters:

    - answer (str): The answer to be included in the humanized response.

    Returns:

    - str: A randomized, humanized response incorporating the given answer.

    Example:
    
    >>> humanized_response("Water boils at 100 degrees Celsius.")
        "Well, after a bit of thought, here's what I found: Water boils at 100
        degrees Celsius."
    """
    # Define a list of response templates that incorporate the answer in a
    # conversational manner.
    humanized_responses = [
        f"Well, after a bit of thought, here's what I found: {answer}",
        f"Ah, got something for you: {answer}",
        f"Here's something: {answer}"
    ]

    # Randomly select one of the response templates to vary the chatbot's
    # responses.
    return random.choice(humanized_responses)
