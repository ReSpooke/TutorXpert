#!/usr/bin/env python3
"""Advanced AI Engineer interviewer with rubric scoring and AI-vs-human response heuristics.

Run:
    python interviewer.py
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from textwrap import fill
import re


@dataclass(frozen=True)
class Question:
    prompt: str
    skill: str
    core_concepts: tuple[str, ...]
    production_concepts: tuple[str, ...]
    follow_up: str


@dataclass(frozen=True)
class AnswerAnalysis:
    normalized: str
    sentence_count: int
    word_count: int
    unique_ratio: float
    ai_likelihood: int
    human_likelihood: int
    classification: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Evaluation:
    score: int
    technical_depth: int
    production_readiness: int
    communication: int
    matched_core: tuple[str, ...]
    matched_production: tuple[str, ...]
    feedback: str


QUESTIONS: tuple[Question, ...] = (
    Question(
        prompt="Design an LLM-powered incident triage assistant for on-call engineers.",
        skill="LLM system design",
        core_concepts=(
            "retrieval",
            "rag",
            "context",
            "prompt",
            "tool",
            "ranking",
            "evaluation",
        ),
        production_concepts=(
            "latency",
            "sla",
            "monitoring",
            "fallback",
            "audit",
            "cost",
        ),
        follow_up="How would you prevent this system from amplifying noisy alerts during incidents?",
    ),
    Question(
        prompt="How would you build a training and deployment lifecycle for a recommendation model?",
        skill="MLOps & lifecycle",
        core_concepts=(
            "feature store",
            "pipeline",
            "validation",
            "ab test",
            "drift",
            "retraining",
        ),
        production_concepts=(
            "ci/cd",
            "versioning",
            "rollback",
            "observability",
            "on-call",
            "governance",
        ),
        follow_up="What leading indicators would tell you to retrain before business KPIs degrade?",
    ),
    Question(
        prompt="You are seeing hallucinations in a regulated domain chatbot. What is your mitigation plan?",
        skill="Reliability & safety",
        core_concepts=(
            "grounding",
            "verification",
            "confidence",
            "guardrail",
            "policy",
            "human review",
        ),
        production_concepts=(
            "escalation",
            "traceability",
            "red team",
            "incident",
            "risk",
            "compliance",
        ),
        follow_up="Which failure mode would you prioritize first, and why?",
    ),
    Question(
        prompt="How do you evaluate model quality when offline and online metrics disagree?",
        skill="Evaluation strategy",
        core_concepts=(
            "precision",
            "recall",
            "calibration",
            "segment",
            "counterfactual",
            "error analysis",
        ),
        production_concepts=(
            "business metric",
            "latency",
            "cost",
            "fairness",
            "reliability",
            "trade-off",
        ),
        follow_up="Describe a concrete trade-off decision you would make in this case.",
    ),
)

AI_STYLE_PHRASES: tuple[str, ...] = (
    "as an ai",
    "in conclusion",
    "it is important to note",
    "delve",
    "furthermore",
    "moreover",
    "comprehensive",
    "robust",
    "leverage",
    "holistic",
)

HUMAN_STYLE_MARKERS: tuple[str, ...] = (
    "i built",
    "in my last role",
    "we shipped",
    "i learned",
    "i would start",
    "i've",
    "can't",
    "won't",
    "didn't",
)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9/+-]+", text.lower())


def split_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
    return sentences


def analyze_origin(answer: str) -> AnswerAnalysis:
    normalized = normalize(answer)
    tokens = tokenize(answer)
    sentences = split_sentences(answer)

    sentence_count = len(sentences)
    word_count = len(tokens)
    unique_ratio = (len(set(tokens)) / word_count) if word_count else 0.0

    ai_likelihood = 0
    human_likelihood = 0
    reasons: list[str] = []

    matched_ai_markers = [phrase for phrase in AI_STYLE_PHRASES if phrase in normalized]
    if matched_ai_markers:
        ai_likelihood += min(35, 6 * len(matched_ai_markers))
        reasons.append(f"AI-style phrasing detected: {', '.join(matched_ai_markers[:4])}")

    matched_human_markers = [phrase for phrase in HUMAN_STYLE_MARKERS if phrase in normalized]
    if matched_human_markers:
        human_likelihood += min(35, 6 * len(matched_human_markers))
        reasons.append(f"Human-like personal markers detected: {', '.join(matched_human_markers[:4])}")

    if sentence_count >= 6 and 14 <= (word_count / max(1, sentence_count)) <= 26:
        ai_likelihood += 12
        reasons.append("Uniformly structured sentence lengths (common in generated text).")

    if unique_ratio < 0.45 and word_count > 70:
        ai_likelihood += 15
        reasons.append("Low lexical diversity for a long answer.")
    elif unique_ratio > 0.62 and word_count > 45:
        human_likelihood += 10
        reasons.append("High lexical diversity suggests non-template phrasing.")

    if re.search(r"\b(uh|hmm|kinda|sort of|maybe)\b", normalized):
        human_likelihood += 10
        reasons.append("Spoken hesitations increase human likelihood.")

    numbered_list_pattern = re.search(r"(^|\n)\s*(1\.|- )", answer)
    if numbered_list_pattern and sentence_count >= 4:
        ai_likelihood += 10
        reasons.append("Neat list formatting is often generated.")

    ai_likelihood = min(95, ai_likelihood)
    human_likelihood = min(95, human_likelihood)

    if ai_likelihood - human_likelihood >= 15:
        classification = "Likely AI-generated"
    elif human_likelihood - ai_likelihood >= 15:
        classification = "Likely human-written"
    else:
        classification = "Uncertain / mixed signals"

    if not reasons:
        reasons.append("Not enough stylistic signals to make a strong origin call.")

    return AnswerAnalysis(
        normalized=normalized,
        sentence_count=sentence_count,
        word_count=word_count,
        unique_ratio=unique_ratio,
        ai_likelihood=ai_likelihood,
        human_likelihood=human_likelihood,
        classification=classification,
        reasons=tuple(reasons),
    )


def evaluate_answer(answer: str, question: Question) -> Evaluation:
    normalized = normalize(answer)

    matched_core = tuple(item for item in question.core_concepts if item in normalized)
    matched_production = tuple(item for item in question.production_concepts if item in normalized)

    technical_depth = min(5, max(1, round((len(matched_core) / len(question.core_concepts)) * 5)))
    production_readiness = min(
        5,
        max(1, round((len(matched_production) / len(question.production_concepts)) * 5)),
    )

    sentence_count = len(split_sentences(answer))
    has_tradeoff_language = bool(re.search(r"\b(trade-off|tradeoff|because|however|risk|constraint)\b", normalized))
    communication = 2
    if sentence_count >= 3:
        communication += 1
    if sentence_count >= 5:
        communication += 1
    if has_tradeoff_language:
        communication += 1
    communication = min(5, communication)

    weighted = round((technical_depth * 0.45) + (production_readiness * 0.35) + (communication * 0.20), 2)
    overall_score = min(5, max(1, round(weighted)))

    feedback_parts: list[str] = []
    if technical_depth >= 4:
        feedback_parts.append("Strong technical depth")
    else:
        feedback_parts.append("Add deeper model/system mechanics")

    if production_readiness >= 4:
        feedback_parts.append("good production awareness")
    else:
        feedback_parts.append("cover operational safeguards (SLA, rollback, monitoring)")

    if communication >= 4:
        feedback_parts.append("clear communication with trade-off framing")
    else:
        feedback_parts.append("be more explicit about trade-offs and constraints")

    feedback = "; ".join(feedback_parts) + "."

    return Evaluation(
        score=overall_score,
        technical_depth=technical_depth,
        production_readiness=production_readiness,
        communication=communication,
        matched_core=matched_core,
        matched_production=matched_production,
        feedback=feedback,
    )


def ask(question_text: str) -> str:
    print(fill(question_text, width=100))
    return input("Your answer: ").strip()


def run_interview() -> None:
    print("\n=== Advanced AI Engineer Interviewer ===")
    print("This simulator scores depth, production readiness, communication, and style-origin signals.")
    candidate_name = input("Candidate name: ").strip() or "Candidate"
    years_exp = input("Years of experience in AI/ML: ").strip() or "N/A"
    print("\nType 'quit' at any time to stop. Recommended answer length: 4-10 sentences.\n")

    evals: list[Evaluation] = []
    origins: list[AnswerAnalysis] = []

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"Q{index} | {question.skill}")
        answer = ask(question.prompt)
        if answer.lower() in {"quit", "exit"}:
            print("\nInterview ended early.")
            break

        evaluation = evaluate_answer(answer, question)
        origin = analyze_origin(answer)

        print(f"Score: {evaluation.score}/5")
        print(
            "Rubric -> "
            f"Depth {evaluation.technical_depth}/5, "
            f"Production {evaluation.production_readiness}/5, "
            f"Communication {evaluation.communication}/5"
        )
        print(f"Matched core concepts: {', '.join(evaluation.matched_core) if evaluation.matched_core else 'None'}")
        print(
            "Matched production concepts: "
            f"{', '.join(evaluation.matched_production) if evaluation.matched_production else 'None'}"
        )
        print(f"Origin classifier: {origin.classification}")
        print(f"Origin confidence snapshot -> AI {origin.ai_likelihood}/95 vs Human {origin.human_likelihood}/95")
        print(f"Feedback: {evaluation.feedback}")
        print(f"Reasoning signal: {origin.reasons[0]}\n")

        evals.append(evaluation)
        origins.append(origin)

        follow_up_answer = ask(f"Follow-up: {question.follow_up}")
        if follow_up_answer.lower() not in {"quit", "exit"}:
            follow_eval = evaluate_answer(follow_up_answer, question)
            print(
                f"Follow-up quick score: {follow_eval.score}/5 "
                f"(Depth {follow_eval.technical_depth}, Production {follow_eval.production_readiness}, "
                f"Communication {follow_eval.communication})\n"
            )
        else:
            print("Skipped follow-up.\n")

    if not evals:
        print("No responses recorded.")
        return

    avg_score = mean(item.score for item in evals)
    avg_depth = mean(item.technical_depth for item in evals)
    avg_production = mean(item.production_readiness for item in evals)
    avg_communication = mean(item.communication for item in evals)
    avg_ai = mean(item.ai_likelihood for item in origins)
    avg_human = mean(item.human_likelihood for item in origins)

    print("=== Final Interview Report ===")
    print(f"Candidate: {candidate_name}")
    print(f"Experience: {years_exp} years")
    print(f"Questions answered: {len(evals)}/{len(QUESTIONS)}")
    print(f"Overall score: {avg_score:.2f}/5")
    print(
        f"Rubric averages -> Depth {avg_depth:.2f}, Production {avg_production:.2f}, "
        f"Communication {avg_communication:.2f}"
    )

    print("\nAnswer-origin signals (heuristic, not definitive):")
    if avg_ai - avg_human >= 12:
        origin_summary = "Responses trend AI-generated"
    elif avg_human - avg_ai >= 12:
        origin_summary = "Responses trend human-written"
    else:
        origin_summary = "Mixed/uncertain origin signals"

    print(f"- Aggregate origin signal: {origin_summary}")
    print(f"- Mean signal scores: AI {avg_ai:.1f}/95 | Human {avg_human:.1f}/95")

    if avg_score >= 4.2:
        recommendation = "Strong AI Engineer readiness. Advance to system design and coding loop."
    elif avg_score >= 3.2:
        recommendation = "Promising profile. Strengthen production trade-offs and reliability depth."
    else:
        recommendation = "Needs additional preparation before a full AI Engineer panel."

    print(f"\nRecommendation: {recommendation}")


if __name__ == "__main__":
    run_interview()
