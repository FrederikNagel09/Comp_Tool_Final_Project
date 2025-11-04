import re

import textstat


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


if __name__ == "__main__":
    sample_text = (
        "This is a sample text. It contains sev eral sentences! "
        "Does it work well? Let's se how it performs."
    )

    print("Word Count:", get_word_count(sample_text))
    print("Character Count:", get_character_count(sample_text))
    print("Lexical Diversity:", get_lexical_diversity(sample_text))
    print("Average Sentence Length:", get_avg_sentence_length(sample_text))
    print("Average Word Length:", get_avg_word_length(sample_text))
    print("Flesch Reading Ease:", get_flesch_reading_ease(sample_text))
    print("Gunning Fog Index:", get_gunning_fog_index(sample_text))
    print("Punctuation Ratio:", get_punctuation_ratio(sample_text))
