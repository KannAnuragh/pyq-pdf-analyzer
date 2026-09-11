import re
import unicodedata
from functools import lru_cache

@lru_cache(maxsize=1)
def load_wordlist(): return frozenset(["മലയാളം"])

@lru_cache(maxsize=1)
def build_bigram_model(): return {}
