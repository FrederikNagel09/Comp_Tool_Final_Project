BATCH_SIZE = 64


NUMERIC_COLS = [
    "word_count",
    "character_count",
    "lexical_diversity",
    "avg_sentence_length",
    "avg_word_length",
    "flesch_reading_ease",
    "gunning_fog_index",
    "punctuation_ratio",
]
STANDARD_COLS = [
    "text",
    "generated",
    "id",
]

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_STATE = 42
N_CLUSTERS_GRID = [2, 30, 100]


NUM_HASH_TABLES_GRID = [16, 24, 32]
NUM_HASH_BITS_GRID = [16, 18]
TOP_K_GRID = [10, 15]
