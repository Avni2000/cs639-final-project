## Training the segmenter - Final Structure
```
segmenter/
├── README.md
├── requirements.txt
├── config.py
│
├── data/                          # prepare and label data
│   ├── __init__.py
│   ├── boundary_detection.py      # split text into reasoning chunks
│   ├── hand_annotate.py           # human labeling: span -> {planning, execution, reflection}
│   ├── llm_judge_labeler.py       # LLM labeling: span -> {planning, execution, reflection}
│   └── build_dataset.py           # convert labeled spans -> train/val jsonl
│
├── model/                         # train and evaluate the segmenter
│   ├── __init__.py
│   ├── classifier.py              # DistilBERT classifier definition - model definition
│   ├── train_segmenter.py         # train classifier on train/val jsonl - train model
│   └── evaluate_segmenter.py      # evaluation + benchmarking - evaluate model performance (optional?)
│
├── inference/                     # run the trained segmenter
│   ├── __init__.py
│   ├── segmenter_pipeline.py      # trace -> spans -> predicted labels {planning, execution, meta-reflection} (model output only)
│   └── probe_selector.py          # select which labeled spans to probe (or call Probe-Baseline) 
│
├── checkpoints/                   # saved model weights
│   └── segmenter/
|
|-- tools/                         # helper scripts (optional)
|   ├── generate_traces.py         # generate additional traces (optional) if needed 
│
└── outputs/                       # generated artifacts
    ├── spans.jsonl                # output of boundary_detection - reasoning chunks
    ├── labeled_spans.jsonl        # output of hand_annotate or llm_judge_labeler - human or LLM annotated spans
    ├── train.jsonl                # output of build_dataset
    └── val.jsonl                  # output of build_dataset

```

*  use </code>`tree segmenter`</code> to view current segmenter layout....

```
segmenter
├── __init__.py
├── 1_data
│   ├── __init__.py
│   ├── boundary_detection.py
│   └── build_dataset.py
├── 2_model
│   ├── __init__.py
│   ├── classifier.py
│   └── train_segmenter.py
├── 3_inference
│   ├── __init__.py
│   ├── probe_selector.py
│   └── segmenter.py
├── 4_checkpoints
│   └── segmenter
├── 5_outputs
│   ├── train.jsonl
│   └── val.jsonl
└── tools
```


##### For Boundary_dectection.py:
> We are using an actual list of “reflection tokens that frequently appear in intermediate reasoning steps” from https://arxiv.org/pdf/2506.12963 (Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills)

```python

# From arXiv:2506.12963 Appendix A.2, plus a few common planning markers.
REFLECTION_TOKENS = {
    "<think>", "wait", "but", "okay", "hmm", "albeit", "however", "yet",
    "still", "nevertheless", "though", "meanwhile", "whereas", "alternatively",
    # Planning/strategy markers (added on top of the paper's list)
    "first", "next", "instead", "therefore", "so", "actually", "let me",
    "let's", "i'll", "i will", "i should", "i need", "another approach",
    "on second thought", "hold on",
}

```


##### For build_dataset.py

This file prepares the dataset for the segmenter.
It takes already-labeled spans and converts them into training and validation files, such as train.jsonl and val.jsonl.

#### Span labeling for the segmenter happens in:

* segmenter/data/hand_annotate.py 
* segmenter/data/llm_judge_labeler.py
* segmenter/inference/segmenter_pipeline.py (prediction stage only)

#### Important note

build_dataset.py does not create the labels itself.
It only organizes spans that have already been labeled.




#### Team members roles (in order of priority):

Jesse:
* segmenter/data/boundary_detection.py - done (i think)
* segmenter/model/classifier.py
* segmenter/inference/segmenter_pipeline.py
* segmenter/inference/probe_selector.py

Bin:
* segmenter/data/hand_annotate.py
* segmenter/data/llm_judge_labeler.py
* segmenter/data/build_dataset.py


Srinivas:
* segmenter/model/train_segmenter.py
* segmenter/model/evaluate_segmenter.py   # optional if time

