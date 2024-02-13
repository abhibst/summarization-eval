import os
import re
import textwrap
import spacy
import numpy as np
from typing import List, Set, Union, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as pairwise_cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
from retry import retry
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from rouge import Rouge
from bert_score import score as bert_score
from typing import List
from pydantic import BaseModel, ValidationError
from loguru import logger
from openai import OpenAI

install(show_locals=False, width=120)
console = Console()


# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Load the OpenAI API key from the environment variables
from dotenv import load_dotenv
load_dotenv()

client_sync = OpenAI()


def aget_embedding(text, model="text-embedding-ada-002", **kwargs):
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    return (
         client_sync.embeddings.create(input=[text], model=model)
        .data[0]
        .embedding
    )


def aget_embeddings(list_of_text, model="text-embedding-ada-002", **kwargs):
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = (
        client_sync.embeddings.create(input=list_of_text, model=model, **kwargs)
    ).data
    return [d.embedding for d in data]


def process_llm_results(generated_text,text,llm_name=""):
    """
    Process the results obtained from the LLM, performing various analysis and comparison steps.

    Args:
    - generated_text (str): The result object from the LLM.
    - text (str): The original text text.
    - llm_name (str): The mode of the current LLM
    """

    rprint("\n")

    try:
        rprint(
            Panel.fit(
                generated_text,
                title="[bold red] Generated Summary [/bold red]",
                subtitle=f"{llm_name}"
            )
        )
    except Exception as e:
        logger.error(f"Error extracting generated text: {e}")
        return

    rprint("\n")  # Add a separator between the tables

    # General Analysis Table
    table = Table(title="General Analysis")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right", style="cyan")

    try:
        overlap_with_original = word_overlap(text, generated_text)
        table.add_row(
            "Word overlap with the original text",
            f"{overlap_with_original * 100:.2f}%",
        )
    except Exception as e:
        table.add_row("Word overlap with the original text", f"Error: {e}")

    try:
        words_original = process_text(text)
        words_summary = process_text(generated_text)
        jaccard_sim = jaccard_similarity(words_original, words_summary)
        table.add_row(
            "Jaccard similarity with the original text", f"{jaccard_sim:.2f}"
        )
    except Exception as e:
        table.add_row(
            "Jaccard similarity with the original text", f"Error: {e}",
        )

    try:
        cosine_sim_tfidf = cosine_similarity_tf_idf(text, generated_text)
        table.add_row(
            "Cosine similarity (TF-IDF based) with the original text",
            f"{cosine_sim_tfidf:.2f}"
            
        )
    except Exception as e:
        table.add_row(
            "Cosine similarity (TF-IDF based) with the original text",
            f"Error: {e}"
            
        )

    # Count of new words in the summary
    try:
        new_words = new_words_in_summary(text, generated_text)
        table.add_row("Count of new words in the summary", str(len(new_words)))
    except Exception as e:
        table.add_row("Count of new words in the summary", f"Error: {e}")

    rprint(table)
    rprint("\n")  # Add a separator between the tables

    # Advanced Analysis Table
    advanced_analysis_table = Table(title="Advanced Analysis")
    advanced_analysis_table.add_column("Metric", style="magenta")
    advanced_analysis_table.add_column("Value", justify="right", style="cyan")
    
    # ROUGE Scores
    try:
        rouge_scores = calculate_rouge_c(generated_text, text)
        for key in rouge_scores[0]:
            advanced_analysis_table.add_row(
                key, f"{rouge_scores[0][key]['f']:.2f}"
            )
    except Exception as e:
        logger.error(f"[bold red]Error computing ROUGE scores:[/bold red] {e}")

    # BERTScore
    try:
        bert_scores = calculate_bert_score(generated_text, text)
        for key, value in bert_scores.items():
            advanced_analysis_table.add_row(
                f"BERTScore {key}", f"{value:.3f}"
            )
    except Exception as e:
        logger.error(f"[bold red]Error computing BERTScore:[/bold red] {e}")
    
    rprint(advanced_analysis_table)   

    # Display new words in summary
    try:
        if new_words:
            rprint(Panel.fit("[bold cyan]New Words in the Summary[/bold cyan]"))
            rprint(Columns([w for w in new_words]))
    except Exception as e:
        logger.error(f"[bold red]Error computing new words in summary:[/bold red] {e}")

    rprint("\n")  # Add another separator

    # Cosine Similarity (Embeddings) Table
    try:
        cosine_sim_pairs = cosine_similarity_embeddings(text, generated_text)
        cos_table = Table(title="Sentence pair cosine-similarity Insights")
        cos_table.add_column("Part of Original Text (Sentence)")
        cos_table.add_column("Part of Generated Summary (Sentence)")
        cos_table.add_column("Score", justify="right", style="cyan")

        # summary_sentences=[s.strip() for s in generated_text.split(".") if s ]
        summary_sentences = [sent.text.strip() for sent in nlp(generated_text).sents]
        top_n = len(summary_sentences)
        for i, pair in enumerate(cosine_sim_pairs[:top_n]):
            if pair['score']>.90:
                cos_table.add_row(
                    pair["original_sentence"],
                    pair["summary_sentence"],
                    f"{pair['score']:.2f}",
                    style="green"
                )
            else:
                cos_table.add_row(
                    pair["original_sentence"],
                    pair["summary_sentence"],
                    f"{pair['score']:.2f}",
                    style="red"
                )  
            cos_table.add_row("", "", "")  # Add an empty row for spacing
        rprint(cos_table)
    except Exception as e:
        rprint(
            f"[bold red]Error computing cosine similarity (embeddings):[/bold red] {e}"
        )



    # # After computing other metrics, display potential hallucinations:
    # try:
    #     hallucinated_sentences = detect_hallucinations(
    #         text, generated_text, manual_threshold=0.3
    #     )

    #     if hallucinated_sentences:
    #         rprint(Panel.fit("[bold cyan]Potential Hallucinated Sentences[/bold cyan]"))

    #         # Create a table to display the hallucinated sentences, their scores, and the threshold
    #         hallucination_table = Table(title="Hallucination Analysis")
    #         hallucination_table.add_column("Sentence", style="magenta")
    #         hallucination_table.add_column("Score", justify="right", style="cyan")
    #         hallucination_table.add_column("Threshold", justify="right", style="green")

    #         for entry in hallucinated_sentences:
    #             hallucination_table.add_row(
    #                 entry["sentence"],
    #                 f"{entry['score']:.2f}",
    #                 f"{entry['threshold']:.2f}",
    #             )

    #         rprint(hallucination_table)
    #         rprint("\n")  # Add a separator
    #     else:
    #         rprint(
    #             "[bold cyan]No potential hallucinated sentences detected.[/bold cyan]"
    #         )
    #         rprint("\n")  # Add a separator
    # except Exception as e:
    #     logger.error(
    #         f"[bold red]Error detecting hallucinated sentences:[/bold red] {e}"
    #     ) 

def get_embed_results(generated_text,text):
    """
    Process the results obtained from the LLM, performing various analysis and comparison steps.

    Args:
    - generated_text (str): The result object from the LLM.
    - text (str): The original text text.
    - llm_name (str): The mode of the current LLM
    """

    # Cosine Similarity (Embeddings) Table
    try:
        cosine_sim_pairs = cosine_similarity_embeddings(text, generated_text)
        cos_table = Table(title="Sentence pair cosine-similarity Insights")
        cos_table.add_column("Part of Original Text (Sentence)")
        cos_table.add_column("Part of Generated Summary (Sentence)")
        cos_table.add_column("Score", justify="right", style="cyan")

        # summary_sentences=[s.strip() for s in generated_text.split(".") if s ]
        summary_sentences = [sent.text.strip() for sent in nlp(generated_text).sents]
        top_n = len(summary_sentences)
        for i, pair in enumerate(cosine_sim_pairs[:top_n]):
            if pair['score']>.90:
                cos_table.add_row(
                    pair["original_sentence"],
                    pair["summary_sentence"],
                    f"{pair['score']:.2f}",
                )
            cos_table.add_row("", "", "")  # Add an empty row for spacing
        rprint(cos_table)
    except Exception as e:
        rprint(
            f"[bold red]Error computing cosine similarity (embeddings):[/bold red] {e}"
        )
    
def preprocess_text(text):
    # Convert to lowercase and remove punctuation and numbers
    text = re.sub(r"[^\w\s]", "", text.lower())

    # Tokenize, lemmatize, and remove stop words
    tokens = [
        token.lemma_ for token in nlp(text) if not token.is_stop and token.is_alpha
    ]

    return " ".join(tokens)


def new_words_in_summary(original_text, summary):
    """
    Identify new words present in the summary that were not in the original text.

    Args:
    - original_text (str): Original text.
    - summary (str): Generated summary.

    Returns:
    - set: Set of new words in the summary.
    """
    original_words = set(
        filter_words([token.text for token in nlp(original_text.lower())])
    )
    summary_words = set(filter_words([token.text for token in nlp(summary.lower())]))

    # Find the new words in the summary
    new_words = summary_words - original_words
    return new_words


def filter_words(word_list):
    """
    Filters out stop words, numbers, and short words from a list of words.

    Args:
    - word_list (List[str]): List of words.

    Returns:
    - Set[str]: Filtered set of words.
    """
    return {
        word
        for word in word_list
        if word not in STOP_WORDS and word.isalpha() and len(word) > 2
    }


def process_text(text):
    """
    Process a given text, tokenizing, lemmatizing, and extracting named entities.

    Args:
    - text (str): Text to be processed.

    Returns:
    - Set[str]: A set of processed words and named entities.
    """
    # Process the text using spaCy
    doc = nlp(text)

    # Extract words and named entities
    words = {
        token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha
    }
    named_entities = {ent.text.lower() for ent in doc.ents}

    # Combine words and named entities
    return words.union(named_entities)


def word_overlap(text1, text2):
    """
    Compute the word overlap between two texts.

    Args:
    - text1 (str): First text.
    - text2 (str): Second text.

    Returns:
    - float: Overlap ratio.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    overlap = words1.intersection(words2)
    return len(overlap) / (len(words1) + len(words2) - len(overlap))


def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard Similarity between two sets.

    Args:
    - set1 (Set[str]): First set.
    - set2 (Set[str]): Second set.

    Returns:
    - float: Jaccard similarity score.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def cosine_similarity_tf_idf(text1, text2) :
    """
    Calculate cosine similarity between two texts using TF-IDF.

    Args:
    - text1 (str): First text.
    - text2 (str): Second text.

    Returns:
    - float: Cosine similarity score.
    """
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return pairwise_cosine_similarity(vectors)[0][1]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def wrap_text(text, width=50, subsequent_indent=""):
    """Wrap text to the specified width."""

    # Break the text into paragraphs (preserving existing line breaks)
    paragraphs = text.split("\n")

    # Wrap each paragraph
    wrapped_paragraphs = [
        textwrap.fill(p, width, subsequent_indent=subsequent_indent) for p in paragraphs
    ]

    # Combine the wrapped paragraphs
    return "\n".join(wrapped_paragraphs)


def cosine_similarity_embeddings(original_text, summary, return_pairs=True):
    """
    Compute the cosine similarity between embeddings of sentences in the original text and the summary.

    This function processes the original text and the summary by splitting them into sentences,
    obtaining embeddings for each sentence, and then computing the cosine similarity
    between each sentence in the original text and each sentence in the summary.

    Parameters:
    - original_text (str): The original text, which will be split into sentences.
    - summary (str): The summary text, which will be split into sentences.
    - return_pairs (bool, optional): If True, returns a list of dictionaries containing
      the pairs of sentences with their cosine similarity score. If False, returns
      only the list of cosine similarity scores. Default is True.

    Returns:
    - Union[List[Dict[str, float]], List[float]]: If return_pairs is True, returns a list of
      dictionaries, each containing 'original_sentence', 'summary_sentence', and 'score'.
      If False, returns a list of cosine similarity scores.
    """
    try:
        # Splitting the original text into sentences
        original_sentences = [sent.text.strip() for sent in nlp(original_text).sents]
        # original_sentences=["CSLY"+r for r in original_text.split("\nCSLY")[1:]]
        summary_sentences = [sent.text.strip() for sent in nlp(summary).sents]
        # summary_sentences=[s.strip() for s in summary.split(".") if s ]
        original_embeddings = aget_embeddings(original_sentences)
        summary_embeddings = aget_embeddings(summary_sentences)

        scores = []

        # Calculate cosine similarity for each pair of sentences
        for j, summ_emb in enumerate(summary_embeddings):
            temp_score=[]
            for i, orig_emb in enumerate(original_embeddings):
                score = cosine_similarity(orig_emb, summ_emb)
                temp_score.append(
                    {
                        # if text is too long, you can use wrap_text to wrap it
                        "original_sentence": (original_sentences[i]),
                        "summary_sentence": (summary_sentences[j]),
                        "score": score,
                    }
                )
            top_1=sorted(temp_score, key=lambda x: x["score"], reverse=True)[0]
            scores.append(top_1)

        # Sort and return the pairs if required
        if return_pairs:
            return scores
            # return sorted(scores, key=lambda x: x["score"], reverse=True)
        else:
            return [score["score"] for score in scores]

    except Exception as e:
        print(f"Error computing cosine similarity (embeddings): {e}")
        return []


# def detect_hallucinations(original_text,summary,method= "average",manual_threshold = None):
#     """
#     Detect potential hallucinated sentences in the summary using semantic similarity.

#     Args:
#     - original_text (str): The original text.
#     - summary (str): The generated summary.
#     - method (str, optional): Method to compute the threshold. Can be "average" or "median".
#     - manual_threshold (float, optional): If set, this threshold will be used instead of dynamically computed thresholds.

#     Returns:
#     - List[dict]: Each dictionary contains the hallucinated sentence, its score, and the threshold used.
#     """
#     scores = cosine_similarity_embeddings(original_text, summary, return_pairs=False)

#     # Compute threshold
#     if manual_threshold:
#         threshold = manual_threshold
#     else:
#         threshold = (
#             sum(scores) / len(scores)
#             if method == "average"
#             else sorted(scores)[len(scores) // 2]
#         )

#     summary_sentences = [sent.text for sent in nlp(preprocess_text(summary)).sents]
#     results = []

#     # Detect hallucinations
#     for idx, score in enumerate(scores):
#         if score < threshold:
#             results.append(
#                 {
#                     "sentence": wrap_text(summary_sentences[idx]),
#                     "score": score,
#                     "threshold": threshold,
#                 }
#             )

#     return results

# based on https://eugeneyan.com/writing/abstractive/
def calculate_rouge_c(summary, document):
    """
    Calculates ROUGE scores comparing a summary with the source document.

    Args:
        summary (str): The generated summary text.
        document (str): The original source document text.

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    rouge = Rouge()
    scores = rouge.get_scores(summary, document)
    return scores


def calculate_bert_score(summary, document):
    """
    Calculates BERTScore comparing a summary with the source document.

    Args:
        summary (str): The generated summary text.
        document (str): The original source document text.

    Returns:
        dict: A dictionary with precision, recall, and F1 BERTScore.
    """
    P, R, F1 = bert_score([summary], [document], lang="en", rescale_with_baseline=True)
    return {"precision": P[0].item(), "recall": R[0].item(), "f1": F1[0].item()}