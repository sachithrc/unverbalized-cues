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



## Aug 13 — Day 2: probe

Split 497/214, stratified, seed 0. C chosen by 5-fold CV on train only.

| detector                        | AUROC |
|---------------------------------|-------|
| shuffled labels (floor)         | 0.466 |
| self-report (leading)           | 0.456 |
| self-report (counterfactual)    | 0.469 |
| self-report (neutral)           | 0.450 |
| bag-of-words on prompt          | 0.643 |
| confidence margin               | 0.749 |
| probe (layer 26)                | 0.961 |
| probe, confidence residualized  | 0.807 |

Confidence direction: LOWER unhinted margin -> more likely to be influenced.

Predicted vs actual: [fill in from your predictions above]

### What I take from this
1. Self-report carries no signal at all — all three framings sit at or below
   the shuffled floor. Not degraded access; none.
2. The information is linearly recoverable from one position of one forward
   pass at 0.961.
3. It is not just uncertainty. Margin alone gets 0.749, so a version of this
   study without that control would have reported mostly recycled confidence.
   Residualizing against margin still leaves 0.807.

### The layer curve — unexpected
Flat ~0.65-0.72 through layer 17, jumps to 0.906 at 18, climbs to 0.961 by 26.
A phase change, not gradual accumulation.

The early plateau (~0.70) tracks the confidence baseline (0.749) almost
exactly. Working hypothesis: early layers encode "this question is hard"
and the deference itself only becomes linearly readable after ~18.

Not yet tested. Would need: probe the early layers against margin directly,
and check whether residualizing collapses layers 0-17 more than 18-27.

### To check
- Rerun with 2-3 other split seeds. 214 test items = roughly +/-0.03 wobble;
  confirm the layer-18 jump isn't a split artifact.
- Shuffled floor is 0.466 not 0.500 — sampling noise at this n, but note it.

### Limitations, running list
- One model (Qwen2.5-1.5B-Instruct), one hint type, one dataset.
- Self-report elicited in the same conversation where the model already
  committed to an answer. Can't distinguish "cannot introspect" from
  "won't contradict itself". The third-person variant would separate these.
- MMLU is in every model's pretraining data.



