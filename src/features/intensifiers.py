"""
Detects "negative-intensifier-as-praise" phrasing -- words that are
lexically negative (so rule-based tools like VADER score them as
negative) but are commonly used colloquially to intensify positive
sentiment, e.g. "disgustingly good", "stupidly entertaining",
"insanely well made".

Why this exists: diagnostic testing showed DistilBERT actually handles
most of this category correctly through contextual understanding, but
on a narrow subset (~3 of 20 tested phrasings, e.g. "disgustingly
addictive", "annoyingly catchy") BERT itself gets it wrong AND is
highly confident AND VADER agrees with the wrong call -- leaving the
meta-model with too little disagreement signal to flag the failure.
More training examples don't fix this efficiently, because VADER's
lexicon will always score these words as negative regardless of
context; what's missing is an explicit signal that the sentence
contains a *known* intensifier-style word in the first place.

This is intentionally a small, hand-curated list rather than an
exhaustive one -- it targets the specific pattern found during
diagnosis, not general sentiment analysis.
"""

NEGATIVE_INTENSIFIER_WORDS = {
    "disgustingly", "stupidly", "ridiculously", "absurdly", "annoyingly",
    "criminally", "sinfully", "painfully", "dangerously", "insanely",
    "frighteningly", "obscenely", "disturbingly", "savagely", "wickedly",
    "scandalously", "recklessly", "shamelessly", "brutally"
}


def has_negative_intensifier(text):
    """
    Returns 1 if the text contains a known negative-intensifier word,
    else 0. Case-insensitive, whole-word match (so "disgusting" alone
    doesn't match "disgustingly", and vice versa won't false-positive
    on unrelated words).
    """
    words_in_text = set(text.lower().split())

    # Strip simple trailing punctuation so "good!" or "good." still
    # matches "good" -- this mirrors how the intensifier words
    # themselves are matched, e.g. "addictive," should still count.
    cleaned_words = {w.strip(".,!?\"'();:") for w in words_in_text}

    return int(bool(cleaned_words & NEGATIVE_INTENSIFIER_WORDS))
