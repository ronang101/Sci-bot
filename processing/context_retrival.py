import functools
import logging
from SPARQLWrapper import SPARQLWrapper, JSON
import wikipedia
from processing.question_answering_logic import answer_question_with_scibert

from utils.utils import score_sentences

# Initialize the SPARQLWrapper with the DBpedia SPARQL endpoint URL.
# This allows for querying DBpedia's vast repository of structured data
# derived from Wikipedia.
sparql = SPARQLWrapper("http://dbpedia.org/sparql")


def fetch_context_for_keyword(result, additional_key_words):
    """
    Attempts to fetch and process relevant context
    for a given keyword by querying Wikipedia.
    This function first retrieves the content of a
    Wikipedia page corresponding to the keyword (result).
    It then scores sentences within this content
    based on their relevance to additional keywords,
    selecting sentences that best match the query.
    This method is useful in this application for
    providing contextually relevant information
    based on user queries is essential.

    Parameters:

    - result (str): The keyword or title of the
      Wikipedia page to fetch content for.
    - additional_key_words (list of str): Additional
      keywords used to score and filter the content for relevance.

    Returns:

    - str: A string composed of relevant sentences extracted from the
      Wikipedia page content, or an empty string if no
      relevant content could be found or an error occurred.

    Exceptions:

    - Logs an error message if fetching content or processing fails.

    Example:

    >>> fetch_context_for_keyword("Climate change", ["global warming",
        "effects"])
        "Relevant sentence 1. Relevant sentence 2. ..."
    """
    try:
        # Fetch the content of the Wikipedia page for the given result/keyword.
        page_content = get_wikipedia_page_content(result)

        # If page content was successfully retrieved
        if page_content:
            # Score and filter the content based on additional keywords to
            # find relevant sentences.
            relevant_sentences = score_sentences(
                page_content, additional_key_words)

            # If relevant sentences were found, return them as a
            # concatenated string.
            if relevant_sentences:
                return (relevant_sentences)

    except Exception as e:
        # Log any errors encountered during the process.
        logging.error(f"Error fetching Wikipedia context for '{result}': {e}")

        return ''


# Decorator to cache results of Wikipedia page content fetches to reduce API
# calls.
@functools.lru_cache(maxsize=100)
def get_wikipedia_page_content(page_title):
    """
    Fetches the content of a Wikipedia page given its title,
    with caching to minimize API requests.
    This function is crucial for applications that rely
    on extracting up-to-date information from
    Wikipedia, such as chatbots or knowledge bases.
    By caching results, it also enhances performance
    and reduces the load on Wikipedia's servers.

    Parameters:

    - page_title (str): The title of the Wikipedia page to retrieve content
      for.

    Returns:

    - str: The content of the Wikipedia page, or None if the page could not be
      found or another error occurred.

    Exceptions:

    - Logs a message for disambiguation or page errors, indicating that the
      page content could not be retrieved.
    """
    try:
        # Retrieve the page content without using Wikipedia's
        # auto-suggest feature.
        page = wikipedia.page(page_title, auto_suggest=False).content
        # Return the retrieved page content.
        return page

    except wikipedia.exceptions.DisambiguationError as e:
        # Log a message if a disambiguation error occurs
        # (multiple pages match the title).
        logging.info(
            f"Disambiguation page found for '{page_title}', skipping...")

    except wikipedia.exceptions.PageError:
        # Log a message if the page could not be found.
        logging.error(f"Page error for '{page_title}', skipping...")

    # Return None if an error occurred and content could not be retrieved.
    return None


# Decorator to cache results of DBpedia fetches to reduce unnecessary queries.
@functools.lru_cache(maxsize=100)
def fetch_from_dbpedia(keyword):
    """
    Retrieves a brief description of a given keyword from DBpedia,
    utilizing SPARQL for querying.
    This function leverages the structured data available in
    DBpedia to provide concise explanations
    or descriptions based on Wikipedia content.
    Caching the results minimizes repetitive queries,
    enhancing performance for applications that
    require quick access to summarized information,
    such as chatbots or automated research tools.

    Parameters:

    - keyword (str): The keyword for which a brief description is sought from
      DBpedia.

    Returns:

    - str: A brief description of the keyword from DBpedia, or an empty string
      if no description was found.

    Exceptions:

    - Does not explicitly handle exceptions but returns an empty string for no
      results, ensuring the application's flow is not interrupted by missing
      data.
    """
    # Set up the SPARQL query to fetch a brief description for the keyword.
    sparql.setQuery(f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?comment
    WHERE {{
      ?s rdfs:label "{keyword}"@en.
      ?s rdfs:comment ?comment.
      FILTER (lang(?comment) = 'en')
    }}
    """)
    # Set the format of the query results to JSON.
    sparql.setReturnFormat(JSON)

    # Execute the query and convert the results to a Python dictionary.
    results = sparql.query().convert()

    # Check if the query returned any results.
    if results["results"]["bindings"]:
        # Extract and return the a single string of all comments found as the
        # description.
        comments = " ".join(binding["comment"]["value"]
                            for binding in results["results"]["bindings"])
        return comments
    else:
        # Return an empty string if no results were found.
        return ''


def fetch_and_process_dbpedia(search_term, question, keywords):
    """
    Fetches a brief description from DBpedia for a given
    search term and processes it to answer a question.
    This function leverages the structured data available
    in DBpedia to obtain concise descriptions,
    which are then scored and used to generate an answer
    to the question using the SciBERT model.
    It aims to utilize semantic web data to enhance the
    chatbot's ability to provide informative responses.

    Parameters:

    - search_term (str): The term for which information is sought from DBpedia.
    - question (str): The user's question that needs answering.
    - keywords (list of str): Keywords extracted from the question to aid in
      scoring sentence relevance.

    Returns:

    - tuple: A tuple containing the answer string and its confidence score,
      or (None, 0) if no relevant context is found.
    """
    # Fetch a brief description for the search term from DBpedia.
    dbpedia_description = fetch_from_dbpedia(search_term)

    # Score and filter the description based on the provided keywords.
    dbpedia_context = score_sentences(dbpedia_description, keywords)

    # If relevant sentences are found in the DBpedia description:
    if dbpedia_context:
        # Use the SciBERT model to answer the question based on the DBpedia
        # context.
        return answer_question_with_scibert(question, dbpedia_description)

    # Return a default response (None) and confidence (0) if no relevant
    # context is found.
    return None, 0


def fetch_and_process_wikipedia(search_term, question, keywords):
    """
    Retrieves and processes content from a Wikipedia page
    matching a given search term to answer a question.
    This function fetches relevant Wikipedia content
    based on the search term, then uses the content to
    generate an answer to the question using the SciBERT
    model. It demonstrates the application's ability
    to integrate real-time web data for generating
    responses, enhancing the user experience by providing
    accurate and contextually relevant answers.

    Parameters:

    - search_term (str): The term for which Wikipedia content is sought.
    - question (str): The user's question that needs answering.
    - keywords (list of str): Keywords extracted from the question to aid in
      scoring sentence relevance.

    Returns:

    - tuple: A tuple containing the answer string and its confidence score,
      or (None, 0) if no relevant context is found.
    """
    # Fetch relevant context from Wikipedia based on the search term and
    # keywords.
    wikipedia_context = fetch_context_for_keyword(search_term, keywords)

    # If relevant context is successfully retrieved:
    if wikipedia_context:
        # Use the SciBERT model to answer the question based on the Wikipedia
        # context.
        return answer_question_with_scibert(question, wikipedia_context)

    # Return a default response (None) and confidence (0) if no relevant
    # context is found.
    return None, 0


def search_and_evaluate_answers(question, accurate_keywords):
    """
    Searches for answers to a given question using a set of
    accurate keywords and evaluates the confidence
    of the responses. This function iterates through
    Wikipedia search results for each keyword and attempts
    to fetch and process relevant information from
    both DBpedia and Wikipedia. Each potential answer is
    evaluated for its confidence level, and the first
    answer meeting the confidence threshold is returned.

    This method demonstrates an integrated approach to
    leveraging multiple data sources for enhancing the
    quality and relevance of chatbot responses, thereby
    improving the user experience by providing accurate
    and trustworthy information.

    Parameters:

    - question (str): The question that needs an answer.
    - accurate_keywords (list of str): Keywords identified
      as most relevant to the question.

    Returns:

    - tuple: A tuple containing the answer and a boolean
      indicating success (True) or failure (False)
      to find a confident answer. Returns (None, False) if no satisfactory
      answer is found.

    Example:

    >>> search_and_evaluate_answers("What is the impact of climate change?",
        ["climate change", "impact"])
        ("Climate change leads to higher sea levels.", True)
    """
    # Iterate through search results level by level, up to 3 levels deep.
    for i in range(3):
        for keyword in accurate_keywords:
            if keyword:  # Ensure the keyword is not empty.
                # Search Wikipedia for pages related to the keyword.
                results = wikipedia.search(keyword)

                # Check if there are enough results to consider the ith result.
                if len(results) > i:
                    result = results[i]

                    # Attempt to fetch and process information from DBpedia
                    # for the ith result.
                    response, confidence = fetch_and_process_dbpedia(
                        result, question, accurate_keywords)
                    # If the response has a high confidence level, return it.
                    if confidence > 0.65:
                        return response, True

                    # If DBpedia does not yield a high-confidence response,
                    # try processing Wikipedia content.
                    response, confidence = fetch_and_process_wikipedia(
                        result, question, accurate_keywords)
                    # Similarly, return the response if it meets the
                    # confidence threshold.
                    if confidence > 0.65:
                        return response, True

    # Return a default response of None and False if no confident answer is
    # found after all attempts.
    return None, False
