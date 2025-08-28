## Results on Webshop

Thanks to [AgentBoard](https://hkust-nlp.github.io/agentboard/static/leaderboard.html)

### Evaluation metrics 
We introduce two primary metrics for evaluation: success rate and progress rate. The success rate measures the proportion of instances in which the goal of an environment is achieved. The progress rate reflects the proportion of completed sub-goals. In addition to these, we incorporate grounding accuracy as a fundamental metric in assessing agent performance, which quantifies the percentage of valid actions taken in each task.

### Public Results

| Model           | Progress Rate (%) | Success Rate (%) | Grounding Accuracy (%) |
|-----------------|-------------------|------------------|-------------------------|
| GPT-4           | 76.5              | 39.0             | 98.3                  |
| Claude2         | 74.6             | 37.8             | 95.9                  |
| GPT-3.5-Turbo   | 76.4             | 35.1             | 90.2                  |
| DeepSeek-67b    | 72.7             | 31.9             | 95.4                  |
| GPT-3.5-Turbo-16k| 73.8            | 27.9             | 96.6                  |
| CodeLlama-13b   | 65.5             | 25.9             | 73.5                  |
| CodeLlama-34b   | 71.7             | 23.5             | 98.3                 |


### Results of Qwen

| Model   | success | num\_actions | reward      | action\_is\_effective | action\_is\_valid | success\_purchase | success\_find | end\_of\_page | r\_type | r\_att      | w\_att      | query\_match | category\_match | title\_score | r\_option   | w\_option   | r\_price    | w\_price    | pass\@1 |
| ------- | --------------- | -------------------- | ------------------- | ----------------------------- | ------------------------- | ------------------------- | --------------------- | --------------------- | --------------- | ------------------- | ------------------- | -------------------- | ----------------------- | -------------------- | ------------------- | ------------------- | ------------------- | ------------------- | --------------- |
| Qwen3-30B-A3B-Thinking-2507 | 0.2421875       | 4.55859375           | 0.12979300726761667 | 0.6789062499999999            | 0.99375                   | 0.2182291666666667        | 0.05182291666666666   | 0.00078125            | 0.19220703125   | 0.13878751240079365 | 0.11897037713443964 | 0.11490885416666666  | 0.18483072916666665     | 0.1060424876634882   | 0.10315755208333333 | 0.06428346899050025 | 0.20963541666666669 | 0.03497532054172679 | 0.2421875       |
|Qwen/Qwen3-30B-A3B| 0.203125        | 4.26953125           | 0.10840565814393939 | 0.684765625                   | 0.9388671875000001        | 0.19205729166666669       | 0.044205729166666666  | 0.0                   | 0.16938802083333335 | 0.11619566902281746 | 0.10348117521945648 | 0.11223958333333334  | 0.16243489583333331     | 0.09257277205539399  | 0.08115234375       | 0.05665500804172679 | 0.19075520833333334 | 0.0319211084054834  | 0.203125        |

