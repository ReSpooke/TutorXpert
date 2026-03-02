# Advanced AI Interviewer for AI Engineer Roles

An enhanced command-line interviewer that evaluates AI Engineer responses across multiple rubric dimensions and estimates whether answers are more likely human-written or AI-generated.

## Run

```bash
python interviewer.py
```

## Features

- Multi-dimensional scoring per answer:
  - Technical depth
  - Production readiness
  - Communication quality and trade-off language
- Follow-up questions after each main prompt.
- AI-vs-human answer-origin heuristics based on style signals.
- Final report with rubric averages, origin trend summary, and recommendation.

## Important note on origin detection

The AI-vs-human detector is heuristic and **not definitive**. It uses stylistic cues (phrasing patterns, lexical diversity, formatting, and personal-experience markers), so it should be treated as a signal rather than proof.

## No external dependencies

This tool uses Python standard library only.
