# Dataset Directory (`data/`)

This directory contains the text datasets used for extracting and evaluating computational subgraphs ($s$-traces) in the paper.

---

## File Format

The dataset files (e.g., `wikitext_40.txt`) are stored in tab-separated value (TSV) format. Each row represents a single evaluation sample formatted as follows:

| Column | Name | Description |
| :--- | :--- | :--- |
| **1** | `input_context` | The prompt sequence consisting of the **first 40 words**. |
| **2** | `ground_truth` | The **41st word** (the ground truth target token). |

Note:  The ground truth next word is not used for faithfulness evaluation (which compares against the full model distribution) but is sometimes used to estimate perplexity of the LLM on the dataset.

---

## Example Entry

```In the absence of Bangladesh's opening bowler, Mortaza, Australia opened the innings with Andrew Symonds and Michael Bevan — giving the usual middle order batsmen time at the crease. The Symonds experiment did not last long  ,```