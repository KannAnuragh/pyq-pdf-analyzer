import os, sys, json, sqlite3
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.abspath(".."))

from pdf_qa_agent.graph.correction_node import correction_and_validation_node

conn = sqlite3.connect('data/pdf_qa.db')
cursor = conn.cursor()
cursor.execute('SELECT result_json FROM jobs WHERE id="fd2c672a-a0d7-44d8-8820-dc172227bb38"')
row = cursor.fetchone()
d = json.loads(row[0])
questions = d.get('questions', [])

state = {
    "extracted_blocks": questions,
    "job_id": "fd2c672a-a0d7-44d8-8820-dc172227bb38",
    "file_path": "fd2c672a-a0d7-44d8-8820-dc172227bb38",
    "pages": [],
}

print("Running correction_and_validation_node on", len(questions), "questions...")
res = correction_and_validation_node(state)
print("Result keys:", res.keys())
print("Summary:", res.get("summary"))
proc = res.get("processed_questions", [])
print("Processed count:", len(proc))
if proc:
    p0 = proc[0]
    print("Q1:", p0.get("question_number"), "type:", p0.get("question_type"), "valid:", p0.get("is_valid"), "val_ans:", p0.get("validated_answer"), "steps:", p0.get("pipeline_steps"))
