## Online serving tests

- Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
- Output length: the corresponding output length of these 200 prompts.
- Batch size: dynamically determined by vllm and the arrival pattern of the requests.
- **Average QPS (query per second)**: 1, 4, 16 and inf. QPS = inf means all requests come at once. For other QPS values, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
- Models: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct
- Evaluation metrics: throughput, TTFT (median time to the first token ), ITL (median inter-token latency) TPOT(median time per output token).

| Test name                        |   Request rate (req/s) |   Tput (req/s) |   Output Tput (tok/s) |   TTFT (ms) |   TPOT (ms) |   ITL (ms) |
|:---------------------------------|-----------------------:|---------------:|----------------------:|------------:|------------:|-----------:|
| serving_qwen2_5_7B_tp1_qps_1     |                      1 |       0.926672 |               207.097 |     94.1281 |     35.2788 |    34.2365 |
| serving_qwen2_5_7B_tp1_qps_4     |                      4 |       2.81837  |               629.863 |     99.4776 |     47.0655 |    40.3403 |
| serving_qwen2_5_7B_tp1_qps_16    |                     16 |       4.48408  |              1002.13  |    109.862  |     73.3124 |    49.7704 |
| serving_qwen2_5_7B_tp1_qps_inf   |                    inf |       5.13334  |              1147.22  |   3082.94   |     64.4224 |    52.5289 |
| serving_qwen2_5vl_7B_tp1_qps_1   |                      1 |       0.993584 |               109.046 |    345.717  |     40.9481 |    32.1247 |
| serving_qwen2_5vl_7B_tp1_qps_4   |                      4 |       3.66915  |               402.579 |    321.537  |     86.7003 |    42.4031 |
| serving_qwen2_5vl_7B_tp1_qps_16  |                     16 |       6.07012  |               668.412 |   8580.35   |    164.858  |    71.5467 |
| serving_qwen2_5vl_7B_tp1_qps_inf |                    inf |       5.26736  |               579.831 |  14326.2    |    208.966  |    73.1931 |

## Offline tests
### Latency tests

- Input length: 32 tokens.
- Output length: 128 tokens.
- Batch size: fixed (8).
- Models: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct
- Evaluation metrics: end-to-end latency.



### Throughput tests

- Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
- Output length: the corresponding output length of these 200 prompts.
- Batch size: dynamically determined by vllm to achieve maximum throughput.
- Models: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct
- Evaluation metrics: throughput.


