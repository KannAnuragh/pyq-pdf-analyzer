import sys
import os
import sqlite3
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.math_normalizer import normalize_math_in_text, heal_units_and_slashes
from ingest.text_cleaner import clean_indic_text

def test_unit_normalization():
    cases = [
        # Option cases
        ("1.5 N C\u2044", "1.5 N/C"),
        ("1.5 N C/", "1.5 N/C"),
        ("1.5 \\times 10^{-16} N C\u2044", "$1.5 \\times 10^{-16}$ N/C"),
        ("3 \\times 10^{-10} N C\u2044", "$3 \\times 10^{-10}$ N/C"),
        ("3 N C\u2044", "3 N/C"),
        ("3 N C/", "3 N/C"),
        ("(d) 3 N C\u2044", "(d) 3 N/C"),

        # Stem cases
        ("Two parallel large thin metal sheets have equal surface densities 26.4 \\times \n10^{-12} C m^2\u2044of opposite signs.",
         "Two parallel large thin metal sheets have equal surface densities $26.4 \\times  10^{-12}$ C/m^2 of opposite signs."),
        ("An infinite line of charge has a linear charge density of 10^{-7} C m\u2044. What will be",
         "An infinite line of charge has a linear charge density of $10^{-7}$ C/m. What will be"),
        ("electric field of 10 V m\u2044. Surface X has a potential of 10 V",
         "electric field of 10 V/m. Surface X has a potential of 10 V"),
        ("that an electric flux of 8 N m^2\u2044 C^{-1} is linked with it.",
         "that an electric flux of 8 N m^2 C^{-1} is linked with it."),

        # Other units
        ("Speed of 60 km h\u2044", "Speed of 60 km/h"),
        ("Velocity is 15 m s\u2044", "Velocity is 15 m/s"),
        ("Pressure in N m^2\u2044", "Pressure in N/m^2"),

        # Non-corrupted units preserved
        ("1.5 N/C", "1.5 N/C"),
        ("Speed is 60 km/h", "Speed is 60 km/h"),
        ("10^{-12} C/m^2", "$10^{-12}$ C/m^2"),
    ]

    print("Running Unit & Slash Normalization Tests...")
    failures = 0
    for original, expected in cases:
        result = normalize_math_in_text(original).strip()
        expected_clean = expected.strip()
        if result != expected_clean:
            print(f"[FAIL]\n  Input:    {repr(original)}\n  Expected: {repr(expected_clean)}\n  Got:      {repr(result)}")
            failures += 1
        else:
            print(f"[PASS] {repr(original)} -> {repr(result)}")

    if failures == 0:
        print("\nAll unit normalization tests PASSED!")
    else:
        print(f"\n{failures} tests FAILED!")
        sys.exit(1)

def test_database_integrity():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pdf_qa.db")
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM jobs WHERE result_json LIKE '%\u2044%'")
    slash_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM jobs WHERE result_json LIKE '%N C/%' OR result_json LIKE '%N C⁄%'")
    nc_count = cursor.fetchone()[0]

    print(f"\nDatabase Verification:")
    print(f"  Jobs with fraction slash (\\u2044): {slash_count}")
    print(f"  Jobs with 'N C/' or 'N C⁄': {nc_count}")

    assert slash_count == 0, f"Expected 0 fraction slashes, found {slash_count}"
    assert nc_count == 0, f"Expected 0 'N C/' occurrences, found {nc_count}"
    print("Database verification PASSED!")

if __name__ == "__main__":
    test_unit_normalization()
    test_database_integrity()
