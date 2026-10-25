# V14 final9029 baseline

Fresh random seed137;500 world-model warmup updates followed by90minutes of policy training on one RTX5090. Final checkpoint9029.

Improved leg separation and left/right symmetry; full backward turn and stable hold. Strict evaluation remains **failed** because of one geometric nonfoot contact frame at2.38s. See README and results/v14.

The checkpoint is our own training output, not an external initialization. Use for evaluation only if claiming to inspect this run; the documented scratch training never loads this release. It retains training metadata (including historical filesystem paths), all networks and optimizer state. Verify SHA256SUMS before loading trusted checkpoint files.

Training source uses pinned unofficial supermimic_vibed. Self-collision disabled as upstream; native CUDA12.8 runtime. Not an official implementation or complete paper reproduction.

An independent packaging-time reevaluation passed with0geometric nonfoot frames and near-identical posture metrics. Both runs are retained: original fail, reevaluation pass. This is evidence of contact sensitivity, not robust repeated success.

`v14_final9029_evaluation.zip` contains the original recorded rollout and200Hz contact-force arrays, original final metrics and the later independent reevaluation metrics. `SHA256SUMS` covers all release artifacts.
