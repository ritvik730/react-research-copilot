# ReAct Research Copilot – Sample Data Pack

This pack contains a **synthetic, test-only** local corpus plus an evaluation set.

## Contents
- corpus/ : synthetic markdown documents + PDFs
- corpus_manifest.json : list of corpus files
- evaluation_questions.csv : 20 questions with gold evidence snippets
- runs_template.csv : logging template for experiments

## Citation Formats
- PDFs: [file:page] (all PDFs are 1 page in this pack)
- Markdown/text: [file] or [file:line] (line numbering optional)

## Suggested Evaluation
- Run each question in two modes:
  1) No-tools baseline (LLM only)
  2) ReAct + tools (retrieval + reading)
- Compute grounded precision: % of cited snippets that support the claim.
