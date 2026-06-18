# H4 cost-benefit ledger

Generated from results\llm_logs (canonical full test-set runs; 'fill' re-runs added to cost totals).

## Headline comparison

| Model | Macro-F1 | $ / 1k predictions | Median latency (s/post) | Est. CO2 (g / 1k predictions) | On-prem deployable |
|---|---|---|---|---|---|
| RoBERTa | 0.6788 | 0 | 0.00162 | 0.058 | Yes (local hardware) |
| MentalBERT | 0.6775 | 0 | 0.00167 | 0.06 | Yes (local hardware) |
| GPT-5.4-mini | 0.6047 | 1.539 | 3.442 | 113.8 | No (cloud API) |
| Gemini 3 Flash | 0.5678 | 0.9797 | 4.019 | 97.97 | No (cloud API) |

## Notes

- `$ / 1k predictions`: actual logged `cost_usd` from the API responses, scaled to 1,000 predictions.
- `median_latency_s`: median wall-clock gap between consecutive logged calls in the canonical run (includes any client-side rate limiting).
- `est_co2_g_per_1k`: rough order-of-magnitude estimate only — 0.30 Wh / 1k tokens x 0.4 kg CO2/kWh (global grid average) for the LLMs; 320 W (GPU) / 65 W (CPU) assumed draw x measured wall time for the local encoders, depending on the benchmark device. Treat as comparable orders of magnitude, not precise figures.
- Grand total LLM spend across **all** logged calls (incl. prompt-iteration / aborted runs): **$5.1712** (budget cap was ~$40-50).
- MentalBERT / RoBERTa run on local hardware already owned by the team: marginal $ cost per prediction is effectively $0.
