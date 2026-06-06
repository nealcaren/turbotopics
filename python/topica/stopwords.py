"""A small English stopword list bundled with topica.

``ENGLISH_STOPWORDS`` is a frozenset of common English function words that rarely
carry topical meaning. Pass it to :func:`topica.tokenize` (or the corpus builders)
so a first LDA fit is not dominated by ``the`` / ``and`` / ``of``::

    docs = [topica.tokenize(t, stopwords=topica.ENGLISH_STOPWORDS) for t in texts]

It is intentionally short and English-only; supply your own list for other
languages or domains.
"""

ENGLISH_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "also",
    "although", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "even", "few",
    "for", "from", "further", "get", "had", "has", "have", "having", "he",
    "her", "here", "him", "his", "how", "however", "if", "in", "into", "is",
    "it", "its", "itself", "just", "many", "more", "most", "much", "must",
    "no", "not", "now", "of", "on", "once", "only", "or", "other", "our",
    "out", "over", "own", "same", "she", "should", "since", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "would", "you", "your",
})
