import en_core_web_trf
import yake
from utils.utils import (clean_keyword,
                   merge_and_boost_keywords,
                   prioritize_compound_terms)


# Load the spaCy model for natural language processing tasks,
# such as stop word filtering.
nlp = en_core_web_trf.load()

# Initialize the YAKE keyword extractor with specific parameters for
# better precision
# lan="en" specifies the language as English
# n=2 sets the maximum number of words in a keyword to 2 (bi-grams)
# dedupLim=0.9 sets the deduplication threshold, keywords with higher
# similarity are deduplicated
# top=3 limits the number of keywords extracted to the top 3 by default
# features=None uses the default features for keyword extraction.
kw_extractor = yake.KeywordExtractor(
    lan="en", n=3, dedupLim=0.8, top=3, features=None)


def extract_keywords(text, num_keywords=5):
    """
    Extracts a specified (max) number of keywords from a
    given text using the YAKE keyword extraction algorithm.
    The function then filters these keywords by removing
    common and domain-specific stopwords to improve
    the relevance of the extracted keywords. This refined
    list of keywords can be used for various natural
    language processing tasks, such as enhancing search
    queries or guiding content analysis in a chatbot.

    Parameters:

    - text (str): The text from which to extract keywords.
    - num_keywords (int): The number of top keywords to
      return after extraction and filtering.

    Returns:

    - list of str: A list of the top N filtered keywords extracted from the
      text.

    Example:

    >>> extract_keywords("Climate change is a global challenge that affects
        everyone everywhere.", 3)
        ["climate change", "global challenge", "affects everyone"]
    """

    # Extract keywords from the given text using YAKE.
    keywords = kw_extractor.extract_keywords(text)

    # Retrieve the default list of stopwords from the spaCy model.
    stopwords = nlp.Defaults.stop_words

    # Define a set of additional stopwords
    # specific to the domain or common queries.
    custom_stopwords = {
        'what', 'explain', 'can', 'you', 'help', 'me', 'understand', 'how',
        'does', 'do', 'why', 'when', 'describe', 'tell', 'definition',
        'example', 'define'
    }

    # Combine the default and custom stopwords to create a comprehensive
    # filter.
    all_stopwords = stopwords.union(custom_stopwords)

    # Filter the extracted keywords, removing any that consist solely of
    # stopwords.
    filtered_keywords = [
        # Use the clean_keyword function to remove stopwords from each keyword.
        clean_keyword(kw[0], all_stopwords)
        for kw in keywords
    ]

    # Return the a max of the top N filtered keywords, as specified by
    # num_keywords.
    return filtered_keywords[:num_keywords]


def extract_nouns_and_entities(text):
    """
    Extracts nouns and named entities from a given text using spaCy,
    aiming to identify key subjects and concepts
    for broader search applicability, such as querying
    Wikipedia pages. Named entities are important as they represent
    specific objects that can be named - such as people,
    places, organizations - while nouns often capture the key
    themes or topics of the text. This dual extraction
    approach ensures a comprehensive set of keywords that can
    enhance information retrieval and analysis processes
    in applications like chatbots.

    Parameters:

    - text (str): The text from which nouns and named entities are to be
      extracted.

    Returns:

    - list of str: A list of unique nouns and named entities extracted from
      the text.

    Example:

    >>> extract_nouns_and_entities("Albert Einstein was a theoretical
        physicist.")
        ["Albert Einstein", "theoretical physicist"]
    """

    # Process the text with the spaCy model to tokenize
    # and analyze its structure.
    doc = nlp(text)

    # Extract named entities directly identified by spaCy.
    entities = [ent.text for ent in doc.ents]

    # Extract nouns by filtering tokens based on their part of speech (POS)
    # tag.
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    # Combine the lists of entities and nouns, ensuring uniqueness by
    # converting to a set and back to a list.
    all_keywords = list(set(entities + nouns))

    # Return the combined and deduplicated list of keywords.
    return all_keywords


def extract_and_process_keywords(question):
    """
    Extracts and processes keywords from a given question by
    utilizing both YAKE and spaCy for keyword extraction,
    then merging and prioritizing the results. This
    comprehensive approach leverages YAKE's ability to identify
    key phrases based on statistical methods and spaCy's
    linguistic insights for extracting nouns and named entities,
    ensuring a rich set of keywords. The process includes
    merging keywords from both methods, boosting shared keywords,
    and prioritizing compound terms, which collectively
    enhance the relevance and accuracy of the keywords for
    subsequent search or analysis tasks in applications such as chatbots.

    Parameters:

    - question (str): The question from which keywords are to be extracted and
      processed.

    Returns:

    - list of str: The top 7 processed and prioritized keywords extracted from
      the question.

    Example:
    
    >>> extract_and_process_keywords("What are the effects of climate change
        on polar bears?")
        ["climate change", "polar bears", "effects"]
    """

    # Extract keywords using YAKE.
    yake_keywords = extract_keywords(question)

    # Extract nouns and named entities using spaCy.
    spacy_keywords = extract_nouns_and_entities(question)

    # Merge and boost keywords from both YAKE and spaCy extractions.
    enhanced_keywords = merge_and_boost_keywords(yake_keywords, spacy_keywords)

    # Prioritize compound terms found within the merged keywords list by
    # putting them to the start of the list.
    accurate_keywords = prioritize_compound_terms(
        enhanced_keywords, question.lower())

    # Limit the number of keywords to the top 7 for focused relevance.
    return accurate_keywords[:7]
