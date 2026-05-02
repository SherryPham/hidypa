# Table: Robustness to paraphrasing attacks (DeepSeek-7B) — local Windows run
# Run from the hidypa\ directory with the venv activated:
#   .\slurm_scripts\run_paraphrasing_attack_deepseek_local.ps1

$ErrorActionPreference = "Stop"

$RUN_TAG = if ($env:RUN_TAG) { $env:RUN_TAG } else { "local_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
Write-Host "Using run tag: $RUN_TAG"
Write-Host "Running paraphrasing attack evaluation (9 configurations) with DeepSeek-LLM-7B"
Write-Host "Paraphrase ratios: 5%, 10%, 15%, 20% x modes: start, middle, end, random"
Write-Host ""

New-Item -ItemType Directory -Force -Path "evaluation/paraphrasing_attack" | Out-Null

$BASE_ARGS = @(
    "--prompts-file", "assets/prompts.txt",
    "--num-prompts", "300",
    "--users-file", "assets/users.csv",
    "--model", "deepseek-llm-7b",
    "--delta", "3.5",
    "--entropy-threshold", "2.5",
    "--hashing-context", "5",
    "--z-threshold", "4.0",
    "--max-new-tokens", "400",
    "--output-dir", "evaluation/paraphrasing_attack",
    "--run-tag", $RUN_TAG
)

# Configuration 1: Naive (L=8)
Write-Host "=========================================="; Write-Host "Configuration 1: Naive (L=8)"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme naive --l-bits 8 @BASE_ARGS

# Configuration 2: Hi-DyPa G=1, U=7
Write-Host ""; Write-Host "=========================================="; Write-Host "Configuration 2: Hi-DyPa G=1, U=7"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme hi_dypa --group-bits 1 --user-bits 7 --l-bits 8 @BASE_ARGS

# Configuration 3: Hi-DyPa G=2, U=6
Write-Host ""; Write-Host "=========================================="; Write-Host "Configuration 3: Hi-DyPa G=2, U=6"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme hi_dypa --group-bits 2 --user-bits 6 --l-bits 8 @BASE_ARGS

# Configuration 4: Hi-DyPa G=3, U=5
Write-Host ""; Write-Host "=========================================="; Write-Host "Configuration 4: Hi-DyPa G=3, U=5"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme hi_dypa --group-bits 3 --user-bits 5 --l-bits 8 @BASE_ARGS

# Configuration 5: Hi-DyPa G=4, U=4
Write-Host ""; Write-Host "=========================================="; Write-Host "Configuration 5: Hi-DyPa G=4, U=4 (paper main config)"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme hi_dypa --group-bits 4 --user-bits 4 --l-bits 8 @BASE_ARGS

# Configuration 6: Hi-DyPa G=5, U=3
Write-Host ""; Write-Host "=========================================="; Write-Host "Configuration 6: Hi-DyPa G=5, U=3"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme hi_dypa --group-bits 5 --user-bits 3 --l-bits 8 @BASE_ARGS

# Configuration 7: Hi-DyPa G=6, U=2
Write-Host ""; Write-Host "=========================================="; Write-Host "Configuration 7: Hi-DyPa G=6, U=2"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme hi_dypa --group-bits 6 --user-bits 2 --l-bits 8 @BASE_ARGS

# Configuration 8: Hi-DyPa G=7, U=1
Write-Host ""; Write-Host "=========================================="; Write-Host "Configuration 8: Hi-DyPa G=7, U=1"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme hi_dypa --group-bits 7 --user-bits 1 --l-bits 8 @BASE_ARGS

# Configuration 9: Group-only G=8, U=0
Write-Host ""; Write-Host "=========================================="; Write-Host "Configuration 9: Group-only G=8, U=0"; Write-Host "=========================================="
python evaluation_scripts/evaluate_paraphrasing_attack.py --scheme hi_dypa --group-bits 8 --user-bits 0 --l-bits 8 @BASE_ARGS

Write-Host ""
Write-Host "=========================================="
Write-Host "All 9 paraphrasing configurations complete!"
Write-Host "Results saved to evaluation/paraphrasing_attack/"
Write-Host "=========================================="
