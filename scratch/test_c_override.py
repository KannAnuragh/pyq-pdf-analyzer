from ingest.option_answer_separator import separate_options_dict

options = {"A": "E != 0 and V != 0", "B": "E = 0 and V = 0", "C": "E != 0 and V = 0"}
marked_answer = "A"
explanation = "(c) E != 0 and V = 0"

clean_opts, ans, exp = separate_options_dict(options, marked_answer, explanation)
print("clean_opts:", clean_opts)
print("marked_answer:", ans)
print("explanation:", exp)
