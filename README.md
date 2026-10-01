# FERMI-2B

**FERMI** (Fast Embedded Reasoning for Machine Inference) is a small open model that reads a **state** (any text or
JSON: an email, a chat message, a log, a board position, a product listing…) and answers **typed questions** about it
with **probabilities**. It never generates text: every answer is a distribution you can threshold, sort or average.

| question type | you give | you get |
|---|---|---|
| `noul` (yes/no) | a question, optional criteria for yes and no | `p_yes` |
| `choice` | a question and 2–26 named options | a probability per option |
| `score` | a question and 2–26 ordered levels | a probability per level + the expected level |

One state can carry many questions; each is scored independently. This is version **0.1**: **MiniCPM5-2B** + a LoRA
adapter + a small decision head, trained on one full epoch of the FERMI data. It is the bigger sibling of
[FERMI-0.8B](https://github.com/Zerflyne/Fermi-0.8B): same code, same demos, same question format, better answers.

## Quick start

```bash
git clone https://github.com/Zerflyne/FERMI-2B && cd FERMI-2B
pip install -r requirements.txt          # PyTorch with CUDA recommended, see Hardware
python examples/quickstart.py
```

```python
from fermi import Fermi

m = Fermi.load()                         # GPU if available, else CPU; downloads openbmb/MiniCPM5-2B once (~5 GB)
m.classify(
    state={"subject": "Charged twice!!", "body": "I was charged twice for March. I need the money back before Friday."},
    questions={
        "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
        "topic":   {"type": "choice", "instructions": "What is the main topic?",
                    "criteria": {"billing": "Payments, charges", "technical": "Bugs, errors", "other": "Anything else"}},
        "urgency": {"type": "score",  "instructions": "How urgent is it?", "criteria": ["Low", "Medium", "High"]},
    })
```

```json
{
 "refund":  {"type": "noul", "answer": "yes", "p_yes": 0.8578},
 "topic":   {"type": "choice", "answer": "billing", "probabilities": {"billing": 0.9936, "technical": 0.0024, "other": 0.004}},
 "urgency": {"type": "score", "answer": 2, "answer_text": "High", "probabilities": [0.0464, 0.1258, 0.8278],
             "expected": 1.781, "expected_01": 0.8907}
}
```

For `noul`, `criteria` is optional: `{"true": "what counts as yes", "false": "what counts as no"}`.
For `score`, list the levels from lowest to highest.

## Interactive demos

```bash
python server.py            # then open http://127.0.0.1:8765
```

- **Playground**: write any state, build your own questions, see every distribution.
- **Live moderation**: FERMI checks a chat message while you type (acceptable? kind? severity?).
- **Inbox triage**: eight emails, three questions each; FERMI sorts them by expected urgency.
- **Twenty questions**: a secret card that FERMI can read and you can't; ask yes/no questions and read the odds.
- **Tic-tac-toe**: play against FERMI; each move is a `choice` over the empty cells, shown as a heat map.

Games that show how much a classifier depends on what you tell it. Most have a **hints** switch: with hints, the page
computes plain facts about each option (this column blocks X, this move hits the wall…) and FERMI still has to read them
and pick; without hints, FERMI gets only the raw board.

- **Snake**: FERMI steers in real time, one "which way?" question per step (17 points in its best game of a short test with hints).
- **Connect four**: play against FERMI and see the probability of every column.
- **Pong**: FERMI holds a paddle at 60 frames per second, answering "up, stay or down?" as fast as it can. In a 38-second
  test against a simple bot it returned 9 balls and missed 1 with hints, returned 2 and missed 8 without.
- **Blackjack**: FERMI decides hit or stand; the textbook basic strategy scores how often it agrees with the math
  (13 of 16 decisions in a short test).
- **Rock, paper, scissors**: FERMI predicts your next move from the history and plays the counter. Against the
  "rock → paper → scissors" bot it predicted 13 of the last 15 moves with hints.

Choice questions in the games are asked with the options in several orders and averaged (`fermiChoice` in
`demos/fermi.js`): small classifiers have a position bias, and this removes most of it.

The server also exposes the model as a JSON API: `POST /api/classify` with `{"state": ..., "questions": {...}}`.

## Hardware

| | memory | speed (measured) |
|---|---|---|
| NVIDIA RTX 4060 8 GB, bfloat16 | ~5 GB VRAM | one question ~30 ms; 24 questions on one state ~0.6 s |
| CPU, float32 | ~11 GB RAM (estimate) | not measured yet |
| Apple MPS | — | should work through the PyTorch path, not tested yet |

`Fermi.load(device="cpu")` forces the CPU. `Fermi.load(base="/path/to/MiniCPM5-2B")` (or `FERMI_BASE=...`) uses a local
copy of the base model instead of downloading it. MiniCPM5-2B is a standard Llama-architecture model: no custom kernels
are needed on any device.

## Results

Evaluation on **5,651 questions about 1,078 states the model never saw** (split by state), the same set used for
FERMI-0.8B. "Teacher" is the model that produced the training labels (see below); agreement = same most-likely answer.

| questions | n | FERMI-2B agreement | FERMI-2B cross-entropy | FERMI-0.8B agreement |
|---|---|---|---|---|
| all | 5,651 | **85.3%** | 0.568 | 81.6% |
| yes/no | 2,389 | 90.6% | 0.326 | 88.7% |
| choice | 1,684 | 85.2% | 0.640 | 81.0% |
| score | 1,578 | 77.4% | 0.857 | 71.6% |

About 0.44 of the cross-entropy is the teacher's own uncertainty, which no student can remove: the real gap to the
teacher (KL divergence) is 0.13.

On a separate set of 200 reference questions answered by an external commercial classifier (Jev), FERMI-2B gives the
same answer **81%** of the time, exactly like the 27B teacher itself (81%); FERMI-0.8B: 76%. Imitating the teacher more
closely will not move this number; better labels would.

We also trained the same recipe on Qwen3.5-2B (half an epoch, not released): 85.3% agreement, cross-entropy 0.578,
80% on the reference questions. MiniCPM5-2B is slightly better on choice and score questions and slightly worse on
yes/no, and its plain architecture runs everywhere without custom kernels.

### Checkpoints

| folder | training step | agreement with the teacher* | agreement on the 200 reference questions* |
|---|---|---|---|
| `checkpoints/step-600` | 600 / 1602 | 82.7% | 78.5% |
| `checkpoints/step-1200` | 1200 / 1602 | 85.3% | 80.0% |
| `checkpoints/final` | 1602 / 1602 | 85.3% (full eval set) | 81.0% |

\* intermediate checkpoints were measured on a 2,538-question subset of the evaluation set.
Each folder holds `adapter.safetensors` (LoRA, bfloat16, ~96 MB), `head.safetensors` and `config.json` (with its
evaluation and the prompt format). Load one with `Fermi.load("checkpoints/step-1200")` or
`python server.py --checkpoint checkpoints/step-1200`.

## How it works

**Input.** State and question are written into one prompt in MiniCPM5's chat format; every option becomes one line
(`A. option: description`). A yes/no question is two lines, `yes` and `no`. Long states are cut to fit 2,048 tokens
(start and end are kept).

**Head.** FERMI reads the last hidden state of every option line and of the final token, which has seen all the
options: `score_i = v · gelu(A·h_option_i + B·h_end + type)`. A softmax over the options gives the answer
distribution. Options are scored by position, so the number of options is not fixed.

**Base and adapter.** [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) (42 layers, 2.5B parameters) is
frozen. A LoRA adapter, rank 32 and alpha 64, sits on the attention and MLP projections of every layer (50.2M
parameters), plus the head (2.1M). At load time the LoRA is merged into the weights, so inference costs the same as the
base model.

**Data.** 48,942 states built from open datasets (emails, chats, reviews, code reviews, papers, recipes, resumes,
server metrics, chess and connect-four positions and more, in several languages), with 252,990 generated questions:
41% yes/no, 31% choice, 28% score.

**Labels.** Soft labels from **Qwen3.8-27B** (NVFP4), asked each question with a clean context: it saw only the state
and the question, and was constrained to one answer token. The label is its renormalized next-token probability.
Choice and score questions were asked twice, the second time with the options in reverse order, and the two
distributions were averaged to cancel position bias. Yes/no labels got a +0.5 logit shift toward "yes", because the
teacher leaned toward "no" on reference questions.

**Training.** Soft-label cross-entropy (a proper scoring rule, so probabilities stay meaningful); AdamW, learning
rate 2e-4 for the LoRA and 1e-3 for the head, 50 warm-up steps then cosine; one full epoch, 247,339 questions
(100M tokens), 64k tokens per step. In training, the options of every choice question were shuffled (labels shuffled
with them), against position bias. It took 4 hours 15 minutes on one RTX 6000 Ada 48 GB.

## Limitations

- **Distilled.** FERMI imitates a 27B teacher and inherits its mistakes; it is not ground truth.
- **Calibration.** FERMI-2B is somewhat overconfident (expected calibration error vs the teacher's top answer: 0.091,
  FERMI-0.8B: 0.068). Treat mid-range probabilities as "unsure", not as exact frequencies. A temperature calibration
  is planned.
- **Weak at games and precise reasoning.** Board positions, arithmetic and multi-step logic are beyond a 2B classifier:
  the game demos show that it plays well only when the page tells it plainly what each move does.
- **Phrasing matters.** Questions that name what the state describes ("Is the product damaged?") work better than
  vague ones ("Is it bad?"). Instructions that list priorities in order work better than long wish lists.
- Not for high-stakes decisions about people (medical, legal, hiring, credit) without human review.

## License

Apache-2.0, like the base model [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B).
