import cProfile
import pstats
from ingest.malayalam_validator import validate_and_repair, load_wordlist, build_bigram_model

# Load models first to avoid profiling IO/init
load_wordlist()
build_bigram_model()

# Some dummy text
text = " ".join(["മലയാളം"] * 1000)

profiler = cProfile.Profile()
profiler.enable()
for _ in range(10):
    validate_and_repair(text)
profiler.disable()

stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats(20)
