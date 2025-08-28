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

| Model   | WebShop/success | WebShop/num\_actions | WebShop/reward      | WebShop/action\_is\_effective | WebShop/action\_is\_valid | WebShop/success\_purchase | WebShop/success\_find | WebShop/end\_of\_page | WebShop/r\_type | WebShop/r\_att      | WebShop/w\_att      | WebShop/query\_match | WebShop/category\_match | WebShop/title\_score | WebShop/r\_option   | WebShop/w\_option   | WebShop/r\_price    | WebShop/w\_price    | WebShop/pass\@1 |
| ------- | --------------- | -------------------- | ------------------- | ----------------------------- | ------------------------- | ------------------------- | --------------------- | --------------------- | --------------- | ------------------- | ------------------- | -------------------- | ----------------------- | -------------------- | ------------------- | ------------------- | ------------------- | ------------------- | --------------- |
| Qwen3-30B-A3B-Thinking-2507 | 0.2421875       | 4.55859375           | 0.12979300726761667 | 0.6789062499999999            | 0.99375                   | 0.2182291666666667        | 0.05182291666666666   | 0.00078125            | 0.19220703125   | 0.13878751240079365 | 0.11897037713443964 | 0.11490885416666666  | 0.18483072916666665     | 0.1060424876634882   | 0.10315755208333333 | 0.06428346899050025 | 0.20963541666666669 | 0.03497532054172679 | 0.2421875       |
