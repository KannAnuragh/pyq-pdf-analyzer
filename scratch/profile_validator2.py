import cProfile
import pstats
from ingest.malayalam_validator import validate_and_repair, load_wordlist, build_bigram_model

load_wordlist()
build_bigram_model()

# Some text not in dictionary, requiring segmentation
# Like "മലയാളം" + "പഠിക്കാം" -> "മലയാളംപഠിക്കാം"
text = " ".join(["മലയാളംപഠിക്കാം"] * 1000)

profiler = cProfile.Profile()
profiler.enable()
for _ in range(1):
    validate_and_repair(text)
profiler.disable()

stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats(20)
