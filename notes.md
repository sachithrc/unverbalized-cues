# Lab notebook

## Aug 12 — predictions (written before any data)

Predicted AUROC for distinguishing hint-influenced from
not-influenced items, within the hinted condition:

- Self-report:          ?
- Unhinted confidence:  ?
- Linear probe:         ?
- J-lens:               ?

What result would surprise me:
What would make me think the design is broken:


## Aug 12 — Day 1: generation

Qwen2.5-1.5B-Instruct, fp16 (T4 = Turing, no bf16).
MMLU: hs_psychology, hs_biology, hs_world_history, marketing.
971 questions -> 711 usable after (a) correct unhinted, (b) unambiguous switch.

- influenced (switched to hint):  270 (38.0%)
- held firm:                      441
- self-report "No" base rate:     20.0%
- activations: (711, 28, 1536), final prompt position

Surprise: the "No" base rate is 20% which is strange because I would've expected the model to
deny influence.

Two candidate explanations:
1. The model is a yes man.
2. The question is leading. "Did the professor's opinion influence your
   answer?" invites Yes. A neutral framing ("what factors determined your
   answer?") would not presuppose the influence.

(2) is a real design flaw and is currently my best candidate for the
"design decision I'd change" application question.

Doesn't invalidate the measurement: the self-report score is continuous
(logit Yes - logit No), so what matters is whether it VARIES with actual
influence, not where it's centred.

Not logged, should have been: unhinted accuracy and the ambiguous-drop
count as separate numbers. Only the combined 73% survival rate is visible.