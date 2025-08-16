## Results on Webshop

Thanks to [AgentBoard](https://hkust-nlp.github.io/agentboard/static/leaderboard.html)

### Evaluation metrics 
We introduce two primary metrics for evaluation: success rate and progress rate. The success rate measures the proportion of instances in which the goal of an environment is achieved. The progress rate reflects the proportion of completed sub-goals. In addition to these, we incorporate grounding accuracy as a fundamental metric in assessing agent performance, which quantifies the percentage of valid actions taken in each task.

### Results

| Model           | Progress Rate (%) | Success Rate (%) | Grounding Accuracy (%) |
|-----------------|-------------------|------------------|-------------------------|
| GPT-4           | 76.5              | 39.0             | 98.3                  |
| Claude2         | 74.6             | 37.8             | 95.9                  |
| GPT-3.5-Turbo   | 76.4             | 35.1             | 90.2                  |
| DeepSeek-67b    | 72.7             | 31.9             | 95.4                  |
| GPT-3.5-Turbo-16k| 73.8            | 27.9             | 96.6                  |
| CodeLlama-13b   | 65.5             | 25.9             | 73.5                  |
| CodeLlama-34b   | 71.7             | 23.5             | 98.3                 |
