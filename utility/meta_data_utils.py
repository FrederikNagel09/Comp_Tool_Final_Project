import re
import textstat

# list of things to count


def get_word_count(text: str) -> int:
    """Total number of words in the text."""
    return len(re.findall(r"\b\w+\b", text))


def get_character_count(text: str) -> int:
    """Total number of characters, including spaces."""
    return len(text)


def get_sentence_count(text: str) -> int:
    """Total number of sentences in the text."""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences)

def get_relative_sentence_count(text: str) -> float:
    """
    Average number of words per sentence in the text.
    """
    word_count = get_word_count(text)
    sentence_count = get_sentence_count(text)

    return word_count / sentence_count


def get_amount_of_paragraphs(text: str) -> float:
    """Total number of paragraphs in the text."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return float(len(paragraphs))

def get_relative_paragraph_count(text: str) -> float:
    """
    Average number of words per paragraph in the text.
    """
    word_count = get_word_count(text)
    paragraph_count = get_amount_of_paragraphs(text)


    return word_count / paragraph_count


def get_amount_syllables(text: str) -> float:
    """Total Number of syllables"""
    return float(textstat.syllable_count(text))

def get_relative_syllable_count(text: str) -> float:
    """
    Average number of syllables per word in the text
    """
    syllable_count = get_amount_syllables(text)
    word_count = get_word_count(text)

    return syllable_count / word_count


def get_lexical_diversity(text: str) -> float:
    """Ratio of unique words to total words (0-1)."""
    words = re.findall(r"\b\w+\b", text.lower())
    return len(set(words)) / len(words) if words else 0.0


def get_avg_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    wc = get_word_count(text)
    sc = get_sentence_count(text)
    return wc / sc if sc else 0.0


def get_avg_word_length(text: str) -> float:
    """Average character length per word."""
    words = re.findall(r"\b\w+\b", text)
    return sum(len(w) for w in words) / len(words) if words else 0.0


def get_punctuation_ratio(text: str) -> float:
    """Ratio of punctuation marks to total characters."""
    puncts = re.findall(r"[^\w\s]", text)
    return len(puncts) / len(text) if text else 0.0


def get_flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease score (0-100, higher = easier)."""
    return textstat.flesch_reading_ease(text)


def get_gunning_fog_index(text: str) -> float:
    """Gunning Fog readability index (grade level required)."""
    return textstat.gunning_fog(text)


def calculate_metadata(text: str) -> dict:
    """Calculate various metadata statistics for the given text."""
    return {
        "word_count": get_word_count(text),
        "character_count": get_character_count(text),
        "lexical_diversity": get_lexical_diversity(text),
        "avg_sentence_length": get_avg_sentence_length(text),
        "avg_word_length": get_avg_word_length(text),
        "flesch_reading_ease": get_flesch_reading_ease(text),
        "gunning_fog_index": get_gunning_fog_index(text),
        "punctuation_ratio": get_punctuation_ratio(text),
    }

def meta_data_vectorrize(text: str) -> list[float]:
    """Return a numeric embedding vector for the text."""

    return [
        get_relative_sentence_count(text),
        get_relative_paragraph_count(text),
        get_relative_syllable_count(text),

        get_lexical_diversity(text),
        get_avg_sentence_length(text),
        get_avg_word_length(text),
        get_punctuation_ratio(text),

        get_flesch_reading_ease(text),
        get_gunning_fog_index(text),
    ]
