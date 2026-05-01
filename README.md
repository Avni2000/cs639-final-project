# cs639-final-project

> Shameless plug that we'll probably eventually see git merge conflicts within Jupyter notebooks, and if you use VSCode, [MergeNB](https://github.com/Avni2000/MergeNB) aims to make the resolution process easy

## Getting Started
- **Model**: [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- **Dataset**: [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024)
- **HDF5 files**: Contain:
  - For every problem `problem_<n>`: 
    - `cot_trace`, which has an error but fixable [cot_trace](cot_trace.md) for an example
       - The error is that the model outputs `</think>` (end tag) but not `<think>` (start tag). So our extraction logic just to get the COT fails. 
          - Can be computed later.
    - `raw_output`, same as COT trace with error; just the raw text output of the whole thing for backup.
    - Hidden States ` hidden_states`: A matrix of shape `number of paragraphs sampled` $\times$ `3584`
       - `number of paragraph sampled` is every paragraph (denoted by `\n\n`) unless there are more than a 100 paragraphs, in which case we cap it at 100 and choose 100 paragraphs at uniform.
          - We grab the hidden states at the last token before the next paragraph.
       - `3584` denotes the fact that each hidden state vector has 3584 elements to it. i.e. it's a list of size 3584.
    - `model_answer` - the answer our grading library thinks our model outputted (eg. 60)
    - `label` - 1 if the `model_answer` aligns up with the aime answer from the dataset, and 0 if it's wrong. (eg. is 60 the right answer?)
    - `truncated` - 1 if model hit max tokens, 8192, and we stopped/truncated its response. 0 if it stopped before then.

**HDF5 files are too big to keep in github**
- I put mine in https://drive.google.com/drive/folders/1WujWT-eqJyFdR9jFdCmtpxj0-7M__nt-?usp=sharing

## Group 1: Segmenter
 - Names: Jesse, Bin, Srinivas
 - See the segmenter folder, figure out what the process will be for training the segmenter. See [proposal](https://docs.google.com/document/d/1dK0CqPyt1aeBTjCyhV2PXzmUZhFl8OiqSEfv2pHNcy4/edit?tab=t.58l62yvrd8rt) + [Jesse's notes on doc](https://docs.google.com/document/d/1dK0CqPyt1aeBTjCyhV2PXzmUZhFl8OiqSEfv2pHNcy4/edit?tab=t.6lhl9j1n63ui)  
   - this'll probably require hand annotating a large dataset, I recommend you use the rest of us too as manpower, and divy up the work equally.

## Group 2: Baseline Probe
 - Names: Avni, Dharini, Rinkle, Gayathri
 - 
## Group 3: Probe
 - TBD
