import sqlite3, json
conn = sqlite3.connect('data/pdf_qa.db')
c = conn.cursor()
c.execute('SELECT created_at, result_json FROM jobs ORDER BY created_at DESC LIMIT 2')
for row in c.fetchall():
    if row[1]:
        data = json.loads(row[1])
        q_count = len(data.get('questions', []))
        print(f"Job created_at: {row[0]}, Questions: {q_count}")
        if q_count >= 5:
            q5 = data['questions'][4]
            print(f"  Q5 type: {q5.get('question_type')}")
            print(f"  Q5 stem starts with: {q5.get('stem', '')[:50]}")
            opts = q5.get('options', {})
            if opts:
                print(f"  Q5 Opt A starts with: {opts.get('A', '')[:50]}")
