from ingest.option_answer_separator import separate_options_dict

opts = {"A": "a", "B": "b"}
exp = "explanation of Assertion. (c) If Assertion is true but Reason is false. (d) If both Assertion and Reason are false. Answer: (d) If both Assertion and Reason are false."

clean_opts, new_marked, new_exp = separate_options_dict(opts, None, exp)
print("clean_opts:", clean_opts)
print("new_exp:", new_exp)
