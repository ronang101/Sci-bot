import re
from nltk.stem import WordNetLemmatizer


def lemmatize_keywords(keywords):
    """
    Converts each keyword in the given list to its base or dictionary form
    (lemma).

    Lemmatization is a process of reducing a word to its base form,
    lemmatization considers the context of a word while doing this.
    This function is particularly useful in natural language
    processing applications where the goal is to analyze or compare
    words at their root level. It helps in standardizing words to
    their root form, which is essential for processes like keyword
    extraction, search, and information retrieval in a chatbot application.

    Parameters:

    - keywords (list of str): A list of keywords to be lemmatized.

    Returns:

    - list of str: A list containing the lemmatized form of the input keywords.

    Example:

    >>> lemmatize_keywords(["running", "leaves"])
        ["run", "leaf"]
    """
    # Initialize the WordNetLemmatizer.
    lemmatizer = WordNetLemmatizer()

    # Prepare a list to hold the lemmatized keywords.
    lemmatized_keywords = []

    # Iterate through each keyword in the provided list.
    for keyword in keywords:
        # Lemmatize the keyword and obtain its base form.
        lemma = lemmatizer.lemmatize(keyword)

        # Append the lemmatized keyword to the list.
        lemmatized_keywords.append(lemma)

    # Return the list of lemmatized keywords.
    return lemmatized_keywords


def clean_keyword(keyword, stopwords):
    """
    Removes stopwords from a keyword phrase, refining it for better processing.

    This function takes a keyword phrase and a list of stopwords, then
    filters out any word within the keyword phrase that is also found
    in the stopwords list. Stopwords are common words that usually have
    little value in the context of search and analysis because they occur
    frequently across all types of texts (e.g., "the", "is", "at"). Removing
    these words from keyword phrases can help improve the relevance and
    focus of search queries and natural language processing tasks, making it a
    crucial step in refining user queries for information retrieval or
    enhancing the quality of keywords extracted for processing in a chatbot
    application.

    Parameters:

    - keyword (str): The keyword phrase to be cleaned.
    - stopwords (set of str): A set of stopwords to filter out from the
      keyword phrase.

    Returns:

    - str: A refined keyword phrase with stopwords removed.

    Example:

    >>> clean_keyword("what is the boiling point", {"is", "the"})
        "boiling point"
    """

    # Split the keyword phrase into individual words and filter out any
    # stopwords.
    return ' '.join(
        word for word in keyword.split()  # Split the phrase into words.
        # Only include words not in the stopwords list.
        if word.lower() not in stopwords
    )


def split_compound_keywords(keywords):
    """
    Splits compound keywords into individual words, filtering out duplicates.

    This function is designed to decompose
    compound keywords into their individual words.
    This is particularly useful in text analysis
    and natural language processing where
    compound keywords might not be recognized
    as a single entity by search or analysis algorithms.
    By breaking down these compounds, the
    function increases the likelihood of matching
    individual words in the context,
    enhancing the chatbot's ability to retrieve
    relevant information based on more granular keyword searches.

    Parameters:

    - keywords (list of str): A list of compound keywords to be split.

    Returns:

    - list of str: A list of unique individual words obtained
      from splitting the compound keywords.

    Example:

    >>> split_compound_keywords(["climate change", "carbon footprint"])
        ["climate", "change", "carbon", "footprint"]
    """

    # Initialize a set to hold the split and unique keywords.
    split_keywords = set()

    # Iterate through each compound keyword in the provided list.
    for keyword in keywords:
        # Split each compound keyword into individual words.
        words = keyword.split()

        # Update the set with the individual words to ensure uniqueness.
        split_keywords.update(words)

    # Return a list of the unique split keywords.
    return list(split_keywords)


def merge_and_boost_keywords(yake_keywords, spacy_keywords):
    """
    Merges keywords extracted by YAKE and spaCy, boosting shared keywords.

    This function takes two lists of keywords
    extracted by different methods (YAKE and spaCy)
    and merges them while giving priority to
    shared keywords. Shared keywords are placed at the
    beginning of the merged list, effectively
    "boosting" their importance. This strategy is
    beneficial in information retrieval and
    natural language processing as it emphasizes keywords
    that are deemed significant by both extraction
    methods, potentially leading to more relevant
    and accurate search results or analysis outcomes
    in the context of a chatbot application.

    Parameters:

    - yake_keywords (list of str): Keywords extracted by the YAKE method.
    - spacy_keywords (list of str): Keywords extracted by spaCy.

    Returns:

    - list of str: A merged and boosted list of unique keywords.

    Example:

    >>> merge_and_boost_keywords(["climate change", "global warming"],
        ["climate change", "environment"])
        ["climate change", "global warming", "environment"]
    """

    # Identify shared keywords for boosting.
    shared_keywords = set(kw for kw in yake_keywords if kw in spacy_keywords)

    # Boost shared keywords by placing them at the start of the list.
    boosted_keywords = list(shared_keywords)

    # Add unique keywords from YAKE extraction not already in the boosted list.
    unique_yake_keywords = [
        kw for kw in yake_keywords if kw not in shared_keywords]
    boosted_keywords.extend(unique_yake_keywords)

    # Add unique keywords from spaCy extraction not already in the boosted
    # list.
    unique_spacy_keywords = [
        kw for kw in spacy_keywords if kw not in shared_keywords and
        kw not in unique_yake_keywords]
    boosted_keywords.extend(unique_spacy_keywords)

    # Return the merged and boosted list of keywords.
    return boosted_keywords


def prioritize_compound_terms(keywords, original_text):
    """
    Prioritizes compound terms (phrases) found in
    the extracted keywords by YAKE, ensuring
    they appear at the beginning of the list.
    This is based on their presence in the original
    text, under the assumption that compound
    terms often carry more specific or nuanced meaning
    than individual words. This method improves the
    relevance of keyword-based searches or analyses
    by highlighting phrases that are likely to be
    more significant in the context of the text.

    Parameters:

    - keywords (list of str): Keywords extracted by YAKE and spaCy
      compound terms (phrases) and single terms.
    - original_text (str): The original text from which the keywords were
      extracted.

    Returns:

    - list of str: A list of keywords with compound terms prioritized at the
      start.

    Example:

    >>> prioritize_compound_terms(["climate change", "climate", "change"],
        "The impact of climate change is significant.")
        ["climate change", "climate", "change"]
    """

    # Identify compound terms by looking for keywords with spaces,
    # and check if they are in the original text.
    compound_terms = [
        kw for kw in keywords if " " in kw and kw.lower() in original_text]

    # Identify single terms as those without spaces.
    single_terms = [kw for kw in keywords if " " not in kw]

    # Combine the lists, placing compound terms first to prioritize them.
    prioritized_keywords = compound_terms + single_terms

    # Return the reordered list of keywords.
    return prioritized_keywords


def score_sentences(page_text, keywords):
    """
    Scores sentences in a given text based on the
    presence and frequency of specified keywords.
    This function is useful for extracting
    relevant sentences from a large text
    based on keyword relevance. It  splits compound
    keywords into individual words
    lemmatizes the keywords to match their base
    forms in sentences, and scores each sentence by
    the number of keyword matches. The sentences
    are then sorted by their scores, and the
    top-scoring sentences are returned, providing
    a concise summary or response based on the keywords.

    Parameters:

    - page_text (str): The text from which
      sentences will be scored and extracted.
    - keywords (list of str): A list of
      keywords used to score the sentences.

    Returns:

    - str: A string composed of the top-scoring sentences, concatenated
      together.

    Example:

    >>> score_sentences("Climate change affects all. It leads to higher
        temperatures.", ["climate change", "temperatures"])
        "Climate change affects all. It leads to higher temperatures."
    """

    # Lemmatize the keywords for uniform matching.
    keywords = lemmatize_keywords(keywords)

    # Split compound keywords into individual words for a broader match.
    keywords = split_compound_keywords(keywords)

    # Replace newline characters with periods to ensure proper sentence
    # splitting.
    page_text = re.sub(r'\n+', '. ', page_text)

    # Split the text into sentences based on punctuation.
    sentences = re.split(r'(?<=[.!?]) +', page_text)

    # Initialize a dictionary to hold sentence scores.
    sentence_scores = {}

    # Score each sentence by counting keyword matches.
    for sentence in sentences:
        # Count how many times any keyword appears in the sentence.
        score = sum(keyword.lower() in sentence.lower()
                    for keyword in keywords)

        # If the sentence contains any keywords, add it to the scoring
        # dictionary.
        if score > 0:
            sentence_scores[sentence] = score

    # Sort the sentences by their scores in descending order.
    sorted_sentences = sorted(sentence_scores.items(),
                              key=lambda x: x[1], reverse=True)

    # Return the top 6 sentences as a single string.
    return ' '.join(sentence for sentence, score in sorted_sentences[:6])


def validate_question(question):
    """
    Validates the user's question based on specific
    criteria to ensure it is in an acceptable format
    for the chatbot to process. This validation includes
    checking the question's length and ensuring
    it begins with certain keywords that typically
    start informational queries. Such validation helps
    maintain the quality of interaction between the
    user and the chatbot, guiding users to phrase their
    questions in a manner that maximizes the likelihood
    of receiving a relevant and accurate response.

    Parameters:

    - question (str): The user's question to be validated.

    Returns:

    - bool: True if the question meets the validation criteria,
      False otherwise.

    The function informs the user of specific reasons
    why their question might be invalid, such as
    exceeding character limits or not starting with an
    approved keyword, and suggests corrections.

    Example:

    >>> validate_question("Why is the sky blue?")
        True
    >>> validate_question("Sky blue why?")
        "Please start your question with one of the
        following words: what, why,
        when, where, explain, define, how, who."
        False
    """
    # Define a set of acceptable start words that are common in informational
    # queries.
    valid_starts = {'what', 'why', 'when',
                    'where', 'explain', 'define', 'how', 'who'}

    # Check if the question length exceeds 100 characters and inform the user
    # if it does.
    if len(question) > 100:
        print("Your question is too long. Please limit it to 100 characters.")
        # Return False to indicate the question does not meet validation
        # criteria.
        return False

    # Extract the first word of the question to check against the set of valid
    # start words.
    # Ensure there is a question to split.
    first_word = question.split()[0].lower() if question else ""

    # Check if the first word is a valid start word and inform the user if it
    # is not.
    if first_word not in valid_starts:
        print(
            f"Please start your question with one of the following words: "
            f"{', '.join(valid_starts)}."
        )

        # Return False to indicate the question does not start appropriately.
        return False

    # If the question passes both checks, return True to indicate it is valid.
    return True

def format_question(question):
    """
    Formats a question by trimming leading/trailing whitespace and ensuring it ends with a question mark.

    Parameters:

    - question (str): The user's input question.

    Returns:
    
    - str: The formatted question.
    """
    # Trim leading and trailing whitespace
    trimmed_question = question.strip()
    
    # Check if the question ends with a question mark, append one if not
    if not trimmed_question.endswith('?'):
        trimmed_question += '?'
    
    return trimmed_question
