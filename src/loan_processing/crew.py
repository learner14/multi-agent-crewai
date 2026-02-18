from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai.tools import tool
import json
import sys
from datetime import datetime
import os


#load the OPENAI_API_KEY from environment variable or set it directly
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = LLM(
    model=os.getenv("MODEL", "gpt-4o"),  # Default to gpt-4o if MODEL env variable is not set
    api_key=OPENAI_API_KEY,  # Or set OPENAI_API_KEY
    temperature=0.0,
    max_tokens=1000,
)



@CrewBase
class LoanProcessing():
    """LoanProcessing crew"""

    #@title Run CrewAI
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def doc_specialist(self) -> Agent:
        return Agent(
                config=self.agents_config['doc_specialist'], # type: ignore[index]
                verbose=True,
                tools=[self.ValidateDocumentFieldsTool],
                llm=llm
            )

    @agent
    def credit_analyst(self) -> Agent:
        return Agent(
                config=self.agents_config['credit_analyst'], # type: ignore[index]
                verbose=True,
                tools=[self.QueryCreditBureauAPITool],
                llm=llm
            )       
    @agent
    def risk_assessor(self) -> Agent:
        return Agent(
                config=self.agents_config['risk_assessor'], # type: ignore[index]
                verbose=True,
                tools=[self.CalculateRiskScoreTool],
                llm=llm
            )
    @agent
    def compliance_officer(self) -> Agent:
        return Agent(
                config=self.agents_config['compliance_officer'], # type: ignore[index]
                verbose=True,
                tools=[self.CheckLendingComplianceTool],
                llm=llm
            )
    @agent
    def manager(self) -> Agent:
        return Agent(
                config=self.agents_config['manager'], # type: ignore[index]
                verbose=True,
                llm=llm,
                allow_delegation=True
            )
    @task
    def task_validate(self) -> Task:
            return Task(
                config=self.tasks_config['task_validate'], # type: ignore[index]
            )

    @task
    def task_credit(self) -> Task:
            return Task(
                config=self.tasks_config['task_credit'], # type: ignore[index]
            )
    @task
    def task_risk(self) -> Task:
            return Task(
                config=self.tasks_config['task_risk'], # type: ignore[index]
            )
    @task
    def task_compliance(self) -> Task:
            return Task(
                config=self.tasks_config['task_compliance'], # type: ignore[index]
            )
    @task
    def task_report(self) -> Task:
            return Task(
                config=self.tasks_config['task_report'], # type: ignore[index]
                allow_delegation=False
            )

    @crew
    def crew(self) -> Crew:
        """Creates the LoanProcessing crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=[self.doc_specialist(), self.credit_analyst(), self.risk_assessor(), self.compliance_officer()], # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            manager_agent=self.manager(), # Automatically created by the @agent decorator
            process=Process.hierarchical,
            verbose=True
        )
    @tool
    def ValidateDocumentFieldsTool( application_data: str) -> str:
        """Validates JSON application data."""
        name: str = "Validate Document Fields"
        description: str = "Validates JSON application data."
        print(f"--- TOOL: Validating document fields ---"+f" (Data: {application_data}) ---")
        try:
            data = json.loads(application_data)
            required = ["customer_id", "loan_amount", "income", "credit_history"]
            missing = [f for f in required if f not in data]
            if missing:
                return json.dumps({"error": f"Missing fields: {', '.join(missing)}"})
            return json.dumps({"status": "validated", "data": data})
        except:
            return json.dumps({"error": "Invalid JSON"})
    @tool
    def QueryCreditBureauAPITool(customer_id: str) -> str:
        """Gets credit score for customer_id."""
        name: str = "Query Credit Bureau API"
        description: str = "Gets credit score for customer_id."
        print(f"--- TOOL: Calling Credit Bureau for {customer_id} ---")
        scores = {
            "CUST-12345": 810, # Good
            "CUST-99999": 550, # BAD SCORE (< 600)
            "CUST-55555": 620
        }
        score = scores.get(customer_id)
        if score:
            return json.dumps({"customer_id": customer_id, "credit_score": score})
        return json.dumps({"error": "Customer not found"})
    @tool
    def CalculateRiskScoreTool(loan_amount: int, income: str, credit_score: int) -> str:
        """Calculates risk based on financial data."""
        name: str = "Calculate Risk Score"
        description: str = "Calculates risk based on financial data."
        print(f"--- TOOL: Calculating Risk (Score: {credit_score}) ---")
        # Logic: Credit Score < 600 is automatic HIGH risk
        if credit_score < 600:
            return json.dumps({"risk_score": 9, "reason": "Credit score too low"})

        # Standard logic
        try:
            inc_val = int(''.join(filter(str.isdigit, income)))
            ann_inc = inc_val * 12 if "month" in income.lower() else inc_val
        except: ann_inc = 0

        risk = 1
        if credit_score < 720: risk += 2
        if ann_inc > 0 and (loan_amount / ann_inc) > 0.5: risk += 3

        return json.dumps({"risk_score": min(risk, 10)})
    @tool
    def CheckLendingComplianceTool(loan_amount: int, risk_score: int) -> str:
        """Checks if loan complies with lending rules."""
        name: str = "Check Lending Compliance"
        description: str = "Checks if loan complies with lending rules."
        print(f"--- TOOL: Checking Lending Compliance ---")
        # Simple compliance logic for demo
        if loan_amount > 500000:
            return json.dumps({"compliant": False, "reason": "Loan amount exceeds limit"})
        if risk_score >= 7:
            return json.dumps({"compliant": False, "reason": "Risk score too high"})
        return json.dumps({"compliant": True})
    