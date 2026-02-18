#!/usr/bin/env python
import sys
import warnings

from datetime import datetime
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from ratelimit import limits, sleep_and_retry
import time

from loan_processing.crew import LoanProcessing


# --- CONFIGURATION ---
CALLS = 15  # Max calls...
PERIOD = 60 # ...per minute

loan_application_inputs_valid = {
    "applicant_id": "borrower_good_780",
    "document_id": "document_valid_123"
}

loan_application_inputs_invalid = {
    "applicant_id": "borrower_bad_620",
    "document_id": "document_invalid_456"
}    


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """

    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }

    try:
        print("--- KICKING OFF CREWAI (VALID INPUTS) ---")
        valid_json = get_document_content(loan_application_inputs_valid['document_id'])
        inputs = {'document_content': valid_json}
        robust_execute(LoanProcessing().crew().kickoff, inputs=inputs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        handle_execution_error(e)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        robust_execute(LoanProcessing().crew().train, n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        handle_execution_error(e)

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        robust_execute(LoanProcessing().crew().replay, task_id=sys.argv[1])

    except Exception as e:
        handle_execution_error(e)

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        robust_execute(LoanProcessing().crew().test, n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        handle_execution_error(e)

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = LoanProcessing().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
    
# --- 1. HELPER: Mock Document Fetcher ---
def get_document_content(document_id: str) -> str:
    print(f"--- HELPER: Simulating fetch for doc_id: {document_id} ---")

    if document_id == "document_valid_123":
        # Happy Path: High Income, Good History
        return json.dumps({
            "customer_id": "CUST-12345",
            "loan_amount": 50000,
            "income": "USD 120000 a year",
            "credit_history": "7 years good standing"
        })

    elif document_id == "document_risky_789":
        # Unhappy Path: Valid Docs, but LOW CREDIT SCORE
        return json.dumps({
            "customer_id": "CUST-99999",
            "loan_amount": 50000,
            "income": "USD 40000 a year",
            "credit_history": "Recent Missed Payments"
        })

    elif document_id == "document_invalid_456":
        # Broken Path: Missing fields (income)
        return json.dumps({
            "customer_id": "CUST-55555",
            "loan_amount": 200000,
            "credit_history": "1 year"
        })
    else:
        return json.dumps({"error": "Document ID not found."})

# --- HELPER: ERROR FILTER ---
def is_rate_limit_error(e):
    msg = str(e).lower()
    return "429" in msg or "quota" in msg or "resource exhausted" in msg or "serviceunavailable" in msg

# --- ROBUST WRAPPER ---
@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception(is_rate_limit_error),
    reraise=True
)
def robust_execute(func, *args, **kwargs):
    """
    Executes any function (CrewAI kickoff, LangGraph invoke) with built-in
    rate limiting and auto-retries for transient API errors.
    """
    print(f"  >> [Clock {time.strftime('%X')}] Executing Agent Action (Safe Mode)...")
    return func(*args, **kwargs)

# --- ERROR HANDLER ---
def handle_execution_error(e):
    """Prints a clean, professional error report."""
    error_msg = str(e)
    is_quota = "429" in error_msg or "quota" in error_msg.lower()

    print("\n" + "━" * 60)
    print("  🛑  MISSION ABORTED: SYSTEM CRITICAL ERROR")
    print("━" * 60)

    if is_quota:
        print("  ⚠️   CAUSE:    QUOTA EXCEEDED (API Refusal)")
        print("  🔍   CONTEXT:  The LLM provider rejected the request.")
        print("\n  🛠️   ACTION:    [1] Wait before retrying")
        print("                  [2] Check API Limits (Free Tier is ~15 RPM)")
    else:
        print(f"  ⚠️   CAUSE:    UNEXPECTED EXCEPTION")
        print(f"  📝   DETAILS:  {error_msg}")

    print("━" * 60 + "\n")

