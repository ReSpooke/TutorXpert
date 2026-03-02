#!/usr/bin/env python3
"""Basic AI interviewer for AI Engineer roles.

Run:
    python interviewer.py
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from textwrap import fill


@dataclass(frozen=True)
class Question:
    prompt: str
    skill: str
    key_points: tuple[str, ...]


QUESTIONS: tuple[Question, ...] = (
    Question(
        prompt="Explain the difference between supervised, unsupervised, and reinforcement learning.",
        skill="ML fundamentals",
        key_points=(
            "labeled",
            "unlabeled",
            "reward",
            "policy",
            "classification",
            "clustering",
        ),
    ),
    Question(
        prompt="How would you design an end-to-end LLM application for customer support?",
        skill="LLM system design",
        key_points=(
            "retrieval",
            "vector",
            "prompt",
            "evaluation",
            "monitoring",
            "guardrail",
            "latency",
        ),
    ),
    Question(
        prompt="What techniques would you use to reduce hallucinations in a production AI system?",
        skill="Reliability & safety",
        key_points=(
            "rag",
            "grounding",
            "verification",
            "confidence",
            "fallback",
            "human",
        ),
    ),
    Question(
        prompt="Describe your approach to MLOps: deployment, monitoring, and retraining.",
        skill="MLOps",
        key_points=(
            "ci/cd",
            "drift",
            "monitoring",
            "feature store",
            "experiment tracking",
            "retraining",
        ),
    ),
    Question(
        prompt="How do you evaluate model quality beyond accuracy?",
        skill="Model evaluation",
        key_points=(
            "precision",
            "recall",
            "f1",
            "auc",
            "latency",
            "cost",
            "fairness",
        ),
    ),
)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def score_answer(answer: str, key_points: tuple[str, ...]) -> tuple[int, list[str]]:
    normalized = normalize(answer)
    matched = [point for point in key_points if point in normalized]
    coverage = len(matched) / max(1, len(key_points))
    score = max(1, min(5, round(coverage * 5)))
    return score, matched


def feedback_for(score: int, missing_count: int) -> str:
    if score >= 4:
        return "Strong answer with good depth."
    if score == 3:
        return "Solid baseline; add more system-level detail and trade-offs."
    if missing_count <= 2:
        return "Good start, but make your answer more structured and specific."
    return "Too high-level. Include concrete methods, metrics, and production constraints."


def run_interview() -> None:
    print("\n=== AI Engineer Interview Simulator ===")
    print("Answer each question in 3-8 sentences. Type 'quit' anytime to end.\n")

    results: list[tuple[Question, int, list[str]]] = []

    for index, question in enumerate(QUESTIONS, start=1):
        print(fill(f"Q{index} ({question.skill}): {question.prompt}", width=95))
        answer = input("Your answer: ").strip()
        if answer.lower() in {"quit", "exit"}:
            print("\nInterview ended early.")
            break

        score, matched = score_answer(answer, question.key_points)
        missing = len(question.key_points) - len(matched)
        feedback = feedback_for(score, missing)

        print(f"Score: {score}/5")
        print(f"Matched concepts: {', '.join(matched) if matched else 'None'}")
        print(f"Feedback: {feedback}\n")

        results.append((question, score, matched))

    if not results:
        print("No responses recorded.")
        return

    average = mean(score for _, score, _ in results)
    print("=== Interview Summary ===")
    print(f"Questions answered: {len(results)}/{len(QUESTIONS)}")
    print(f"Average score: {average:.2f}/5")

    print("\nSkill breakdown:")
    for question, score, matched in results:
        print(
            f"- {question.skill}: {score}/5 "
            f"({len(matched)}/{len(question.key_points)} key concepts covered)"
        )

    if average >= 4:
        recommendation = "Strong fit for an AI Engineer interview loop."
    elif average >= 3:
        recommendation = "Promising profile; strengthen architecture and production-depth answers."
    else:
        recommendation = "Needs improvement before high-stakes AI Engineer interviews."

    print(f"\nOverall recommendation: {recommendation}")


if __name__ == "__main__":
    run_interview()
