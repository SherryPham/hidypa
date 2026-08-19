# Segment-WM baseline — one extra row for Tables II–XIII

**What this adds.** The segment-assignment multi-bit watermark from *Provably Robust
Multi-bit Watermarking for AI-generated Text* (the `segment-wm/` repository, "Segment-WM"
below) is now available as `--scheme segment` in every evaluation script that produces
Tables II–XIII. It is a **flat 8-bit scheme** — a user's codeword is the binary expansion
of their user ID — so it reports in the paper exactly like the MAU baseline, at
`(Lg, Lu) = (0, 8)`.

Everything else is held fixed against the existing rows: same model, same
`assets/prompts.txt`, same 128 users, same 300 prompts, same attacks and intensities,
same metric definitions, and the same tracing / merging / feasible-set code paths that
MAU already uses.

---

## 1. Which script produces which table

| Table | Experiment | Script | SLURM script |
|---|---|---|---|
| II | Clean-text detection | `evaluate_hi_dypa_detection.py` | `run_segment_detection_hpc.sh` |
| III, IV | Collusion (2 and 3 colluders) | `compare_collusion_resistance.py` | `run_segment_collusion_hpc.sh` |
| V | Framing attack (CRR at k = 50) | `evaluate_framing_attack.py` | `run_segment_framing_hpc.sh` |
| VI (GPT-2), VII (DeepSeek) | Token deletion | `evaluate_hi_dypa_robustness.py` | `run_segment_robustness_hpc.sh` |
| VIII (GPT-2), IX (DeepSeek) | T5 paraphrasing | `evaluate_paraphrasing_attack.py` | `run_segment_paraphrasing_hpc.sh` |
| X (GPT-2), XI (DeepSeek) | WordNet synonym substitution | `evaluate_synonym_attack.py` | `run_segment_synonym_hpc.sh` |
| XII (GPT-2), XIII (DeepSeek) | LLM rewriting | `evaluate_rewrite_attack.py` | `run_segment_rewrite_hpc.sh` |
| XIV | Tracing time (ms) | `evaluate_multiuser_performance.py` | `run_segment_performance_hpc.sh` |

Tables II, III, IV, V and XIV each carry a GPT-2 *and* a DeepSeek-7B column, so those
scripts are run twice (`MODEL=deepseek-llm-7b` for the second pass). Tables VI–XIII are
split one model per table.

Each SLURM script runs **only** the segment configuration, so none of the existing
MAU / Hi-DyPa results are recomputed or overwritten. Results land in
`evaluation/<experiment>/segment/L8/<run_tag>/`, alongside the existing
`naive/L8/` and `hi_dypa/G*_U*/` directories.

---

## 2. How the scheme works here

New modules:

| File | Contents |
|---|---|
| `src/reedsolomon.py` | Reed-Solomon over GF(2^m) with no external dependencies (the reference implementation needs `galois`, which is not in `hidypa.sif`). Includes an exhaustive nearest-codeword decoder. |
| `src/segment_watermark.py` | `SegmentWatermarker` (embedding + detection) and `SegmentMultiUserWatermarker` (flat tracing, a subclass of `NaiveMultiUserWatermarker`). |
| `src/segment_cli.py` | The shared `--segment-*` argparse group and construction helper used by all seven evaluation scripts. |
| `helper_scripts/build_segment_bh_map.py` | Builds the frequency-balanced token → segment map for the `rsbh` variant. |
| `helper_scripts/collect_segment_rows.py` | Turns the result JSONs into paste-ready table rows. |

**Embedding.** The 8-bit payload is split into `k = 2` symbols of `m = 4` bits and
Reed-Solomon encoded into `n = 4` symbols over GF(16) — `RS(4, 2)`, correcting one symbol
error. At each generated position the previous token seeds a PRNG that draws a green list
of `gamma · V` tokens; the green bias vector is cyclically shifted by the symbol assigned
to that position, and `delta` is added to the shifted green tokens.

**Detection.** The detector recomputes the green list at every position, accumulates a
`COUNT` array of size `2^m` per segment, takes the arg-max per segment, and decodes the
resulting `n` symbols back to a payload. The payload is the user ID.

### Three deliberate deviations from the reference code

These are all documented so they can be stated in the paper; each one makes the
comparison *more* favourable to the baseline or more directly comparable, never less.

1. **Exhaustive nearest-codeword decoding instead of bounded-distance syndrome
   decoding.** With 8 bits there are only 256 codewords, so the maximum-likelihood
   decoder is cheap. It corrects everything the syndrome decoder corrects and, in
   addition, returns the most likely payload instead of failing when more than
   `t = 1` symbol is wrong. Pass `--segment-syndrome-decode` to use the reference
   (weaker) decoder instead.

2. **The detector never abstains (default), and reports a null-centred z-score.**

   *No abstention.* The reference decoder always emits a payload, and so does this one
   by default (`--segment-z-threshold -1`). For this row, therefore,
   **FP = 1 − Acc and FN = 0** — worth one sentence in the paper, since MAU and Hi-DyPa
   can return `⊥` and abstain. This is the conservative choice: it does not handicap
   the baseline's accuracy, and it makes Hi-DyPa's lower false-positive rate a
   consequence of hierarchical gating rather than of a threshold we picked for the
   competitor.

   *Why not gate.* An abstention threshold was implemented and measured, and it costs
   the baseline accuracy for no benefit. On 12 GPT-2 prompts at 512 tokens the decoder
   recovered **12/12** payloads correctly, but gating at z ≥ 4 would have rejected 3 of
   them. The cause is degenerate repetition: GPT-2 loops, so one 515-token generation
   contained only 43 distinct `(context, token)` pairs. The evidence is genuinely
   low-rank, yet the repeated tokens all vote for the correct symbol, so the decoder is
   right while the statistic is (correctly) unconvinced. Turning correct attributions
   into abstentions would understate the baseline. To evaluate the abstaining variant
   anyway, pass `--segment-z-threshold 4.0`; the z-score is recorded per prompt either
   way, so any threshold can also be applied post hoc to `raw_results.jsonl.gz`.

   *Null-centred statistic.* The reference z-score is
   `(Σ_i max(COUNT_i) − γT) / sqrt(γ(1−γ)T)`, which ignores the upward bias of taking a
   maximum over `2^m` candidates per segment. Measured on GPT-2, **unwatermarked** text
   scores a mean of 5.2 on that statistic, so a threshold of 4.0 would mean nothing.
   `--segment-z-mode calibrated` (default) instead subtracts the exact null mean of the
   per-segment maximum and divides by its null standard deviation, both computed from
   the binomial distribution, over de-duplicated `(context, token)` pairs. On 25
   unwatermarked GPT-2 samples that statistic has mean 0.17 and sd 1.21 (max 2.23),
   versus a mean of 8.5 on watermarked text. The **payload is still decoded from all
   positions**, so no watermark signal is lost. `--segment-z-mode raw` reports the
   reference formula instead.

3. **Segment assignment defaults to `rs`, not `rsbh`.** The paper's headline variant
   (`rsbh`) assigns positions to segments through a frequency-balanced hash of the
   previous token, which needs a per-model token-frequency table and a compiled DP
   helper. The `rs` variant draws the assignment from the same context-seeded PRNG,
   needs no auxiliary artefact, and is uniform in expectation. To run the balanced
   variant instead:

   ```bash
   python helper_scripts/build_segment_bh_map.py \
       --model gpt2 --segments 4 \
       --corpus assets/prompts.txt \
       --output assets/segment_bh_map_gpt2_n4.json

   # then add to any evaluation command:
   --segment-assign rsbh --segment-bh-map assets/segment_bh_map_gpt2_n4.json
   ```

   The same map file must be used for generation and detection, so build it once per
   `(model, n)` pair.

### Parameters and why they are set that way

| Parameter | Default | Rationale |
|---|---|---|
| `--segment-rs 4 2 4` | `RS(4,2)` over GF(16) | `k · m` must equal `L = 8`. The authors' `rs_search.py` only covers 12–32 bits and its rate constraint (`k/n ≥ 0.6`) is unsatisfiable at 8 bits, so this is the smallest even-redundancy code that still corrects one symbol error. |
| `--segment-gamma 0.5`, `--segment-delta 6.0` | Segment-WM paper values | `delta` is **not** comparable to Hi-DyPa's `--delta`, which scales a Gaussian score vector rather than boosting a green list. Each scheme runs at its own recommended operating point. |
| `--segment-ngram 1` | Segment-WM paper value | Context width for the green-list seed. |
| `--segment-temperature 0.7`, `--segment-top-p 0.95`, `--segment-top-k 50` | Hi-DyPa pipeline values | Sampling is held **identical** to the other rows so that the only difference between rows is the watermark mechanism. Use `--segment-temperature 1.0 --segment-top-k 0` for the Segment-WM paper's own sampling. |
| `--segment-suppress-eos` | off | The reference implementation forces full-length generations by suppressing EOS. Left off so generation terminates exactly like the other rows. |
| `--segment-z-threshold -1` | never abstain | See deviation 2. Use `4.0` for the abstaining variant. |

**Caveat worth stating in the paper:** `delta = 6.0` is a strong green-list bias and this
evaluation measures attribution accuracy only, not text quality. If a reviewer asks about
the quality/robustness trade-off, a perplexity comparison would need to be run separately —
it is not part of Tables II–XIII.

---

## 3. Verification already done

* `RS(n, k, m)` encode/decode round-trips and corrects exactly `t` symbol errors for
  `(4,2,4)`, `(6,2,4)`, `(8,2,4)`, `(3,1,8)`, `(6,4,8)`, `(14,8,4)`, `(15,9,4)` —
  400/400 trials each.
* End-to-end embed → detect round-trip on GPT-2 recovers the payload exactly on clean
  text: 12/12 payloads at 512 tokens, 23/25 on a separate 200-token batch.
* Null-distribution measurement on unwatermarked GPT-2 text (see deviation 2 above).
* All seven evaluation scripts run to completion with `--scheme segment` and write
  `summary.json` with the expected metric keys plus a `segment_config` provenance block.

Not yet run: the full 300-prompt jobs, which is what the SLURM scripts below are for.

---

## 4. Reading the results

```bash
python helper_scripts/collect_segment_rows.py --evaluation-dir evaluation --model gpt2 --latex
```

prints one row per table in the paper's column order, plus the LaTeX line. Run it again
with `--model deepseek-llm-7b` to get the Table VII / IX / XI / XIII numbers.
