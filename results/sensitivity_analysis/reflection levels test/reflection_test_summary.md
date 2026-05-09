# Reflection levels sensitivity — summary

**Source:** `summary_reflection_levels_260504_004004.json` and each scenario’s `results_*.json` under `results/sensitivity_analysis/reflection levels test/`.

Fixed settings match `results/sensitivity_analysis/reflection levels test/config.py`, except `**REFLECT_MODEL_NAME`** and `**REFLECTION_LEVELS`** per scenario.

**Problem:** `problem_data/small/small_k1.5_instance_001.json` (from template config at run time).

## Results (run order)


| #   | `REFLECT_MODEL_NAME`            | `REFLECTION_LEVELS` | Final tardiness | Max tardiness | Iterations | Runtime (s) | Session name                                                |
| --- | ------------------------------- | ------------------- | --------------- | ------------- | ---------- | ----------- | ----------------------------------------------------------- |
| 1   | `google/gemini-3-flash-preview` | 0                   | 26.202          | 7.142         | 19         | 527.274     | `260504_004004_reflectlvl_google_gemini_3_flash_preview_L0` |
| 2   | `google/gemini-3-flash-preview` | 1                   | 27.549          | 9.204         | 19         | 602.174     | `260504_004004_reflectlvl_google_gemini_3_flash_preview_L1` |
| 3   | `google/gemini-3-flash-preview` | 2                   | **24.442**      | 7.295         | 19         | 779.929     | `260504_004004_reflectlvl_google_gemini_3_flash_preview_L2` |
| 4   | `google/gemini-3-flash-preview` | 4                   | 27.549          | 9.204         | 19         | 701.204     | `260504_004004_reflectlvl_google_gemini_3_flash_preview_L4` |
| 5   | `google/gemini-3.1-pro-preview` | 0                   | 27.549          | 9.204         | 19         | 765.279     | `260504_004004_reflectlvl_google_gemini_3_1_pro_preview_L0` |
| 6   | `google/gemini-3.1-pro-preview` | 1                   | 27.549          | 9.204         | 19         | 776.159     | `260504_004004_reflectlvl_google_gemini_3_1_pro_preview_L1` |
| 7   | `google/gemini-3.1-pro-preview` | 2                   | 27.549          | 9.204         | 19         | 819.796     | `260504_004004_reflectlvl_google_gemini_3_1_pro_preview_L2` |
| 8   | `google/gemini-3.1-pro-preview` | 4                   | 26.202          | 7.142         | 19         | 1162.994    | `260504_004004_reflectlvl_google_gemini_3_1_pro_preview_L4` |


**Best (lowest final tardiness):** `REFLECT_MODEL_NAME=google/gemini-3-flash-preview`, `REFLECTION_LEVELS=2` → final tardiness **24.442** (runtime **779.929 s**).

**Tie (second place, 26.202):** flash `L0` and pro `L4` (pro `L4` has higher runtime **1162.994 s**).

## Same data sorted by final tardiness (then runtime)


| `REFLECT_MODEL_NAME`            | `REFLECTION_LEVELS` | Final tardiness | Runtime (s) |
| ------------------------------- | ------------------- | --------------- | ----------- |
| `google/gemini-3-flash-preview` | 2                   | 24.442          | 779.929     |
| `google/gemini-3.1-pro-preview` | 4                   | 26.202          | 1162.994    |
| `google/gemini-3-flash-preview` | 1                   | 27.549          | 602.174     |
| `google/gemini-3-flash-preview` | 4                   | 27.549          | 701.204     |
| `google/gemini-3.1-pro-preview` | 1                   | 27.549          | 776.159     |
| `google/gemini-3.1-pro-preview` | 2                   | 27.549          | 819.796     |


