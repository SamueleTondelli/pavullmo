# Context–target accuracy gap and score-matrix diagnostics

Analysis of the saved synthetic benchmark reports for `35m_balanced_1b5`.
There are 30 benchmarks in six categories, with 100 contrast sets per
benchmark. The analysis uses the detailed reports already written under
`src/evaluation/benchmarks/scores/`; it does not rescore the model.

## Definitions

For one contrast set, let `S[i,j]` be the **mean target-token log probability**
of target `j` after context `i`. Rows are contexts, columns are targets, and
the correct matches are on the diagonal. The model never sees a displayed
multiple-choice list; the alternatives are scored separately.

- Context accuracy fixes target `j` and asks whether `S[j,j] > S[i,j]` for
  each other context `i`. The scorer actually uses total target log
  probability here, but target length is fixed within a column, so the
  ranking is identical.
- Target accuracy fixes context `i` and asks whether `S[i,i] > S[i,j]` for
  each other target `j`. The scorer uses mean target-token log probability.
- The accuracy gap is context accuracy minus target accuracy. The gap script
  defines overall benchmark accuracy as the mean of those two accuracies.
- Within a set, variance across contexts is the average, over targets, of
  the population variance down each column. Variance across targets is the
  average, over contexts, of the population variance across each row. Both
  use the same per-token score scale.

For a square score matrix, the additive decomposition is
`S[i,j] = grand_mean + context_effect[i] + target_effect[j] + interaction[i,j]`.
The two conditional variances satisfy:

```text
variance_across_contexts = variance(context_effect) + variance(interaction)
variance_across_targets  = variance(target_effect)  + variance(interaction)
```

Thus a larger across-target variance directly indicates a larger target
baseline effect than context baseline effect. It does not by itself show
whether the context–target interaction identifies the correct answer.

Column centering subtracts the mean score for each target across **all**
contexts in its set: `S_centered[i,j] = S[i,j] - mean_k(S[k,j])`. Target
pairwise accuracy is recalculated with strict wins, then averaged over sets,
benchmarks, and categories. Centering leaves context ranking for any fixed
target unchanged. It uses the whole contrast set, so the result is a
**diagnostic**, not an accuracy available for an isolated prompt. The same
comparison is plotted by `plot_centered_tgt_scores(model_name)` in
`src/evaluation/evaluate_pretrain.ipynb`.

## Findings

- The benchmark-level mean-accuracy/gap association is strong: Pearson
  `r = 0.844`, Spearman `rho = 0.896`, and within-category Pearson
  `r = 0.773`. Five benchmarks have mean accuracy below 0.5; all five have
  negative gaps. This is descriptive: accuracy and gap contain the same two
  component scores. In fact,
  `Cov((context + target)/2, context - target) =
  (Var(context) - Var(target))/2`.
- Across the 3,000 sets, mean score variance across targets is `0.9186`,
  compared with `0.1839` across contexts, a `5.00×` ratio. Target variance
  is higher in 97.1% of sets and in all 30 benchmarks. Target baseline
  effects account for 80.6% of the within-set score variance; the
  context–target interaction accounts for 16.5%.
- ICL is the clearest case: the variance ratio is `49.41×`, target baseline
  effects account for 98.0% of variance, and the same target ranks first
  in every context in 86.8% of sets. Raw target pairwise accuracy is 51.1%
  and correct top-target accuracy is 25.6%, both near chance. Yet the
  matched-pair interaction lift is positive in 82.6% of sets. Small
  context-sensitive differences can therefore support context comparison
  without reliably overturning target preferences.
- Column centering raises ICL target pairwise accuracy from 51.1% to 66.8%
  and overall target pairwise accuracy from 58.5% to 68.1%. It does not
  help every category: reasoning falls from 44.8% to 39.0%, consistent with
  its negative mean matched-pair interaction lift. A deployable calibration
  method needs an independently available baseline and held-out validation.
- Target index 0 ranks first in 38.1% of ICL contexts versus a 25% uniform
  index baseline. The ICL configurations fix query indices at `0,1,2,3`,
  so this is a possible ordering effect, not proof of one. Shuffling the
  binding order while holding the mapping and query fixed would test it.

The full output below gives all category values, including median score
margins, top-target collapse rates, and interaction lifts.

## Full command output

The following blocks are the complete stdout of the commands shown, run from
the project root. Neither command produced stderr.

### `python src/evaluation/analyze_accuracy_gap.py`

```text

35m_balanced_1b5: 30 benchmarks across 6 categories
Accuracy = (context + target) / 2; gap = context - target
Benchmark Pearson r=+0.844, permutation p=0.0001
Benchmark Spearman rho=+0.896, permutation p=0.0001
Within-category Pearson r=+0.773, permutation p=0.0019
Category means (descriptive; five benchmarks per category):
  category                 n   accuracy       gap
  icl                      5   0.559      +0.095
  knowledge                5   0.701      +0.116
  language-manipulation    5   0.902      +0.162
  math                     5   0.621      +0.076
  math-languages           5   0.503      -0.002
  reasoning                5   0.416      -0.064
  Category-mean Pearson r=+0.906
Chance split, using each benchmark's mean accuracy:
  below 0.5    n= 5, mean gap=-0.076, positive=0, negative=5
  at/above 0.5 n=25, mean gap=+0.092, positive=21, negative=4
Context variance=0.0517; target variance=0.0203
Interpretation: mean accuracy and gap share the same two scores; their correlation is descriptive, not independent evidence that higher model quality causes a larger gap.
Permutation p-values describe this collection of benchmarks and assume exchangeable benchmarks within each shuffle group.
```

### `python src/evaluation/analyze_score_matrices.py`

```text
Variances use mean log probability per target token (population variance).
Target index means saved contrast-set order, not a choice list in the prompt.
Some benchmarks fix member order in their templates, so index 0 frequency alone does not prove a position bias.

=== 35m_balanced_1b5: 30 benchmarks ===

Overall: 3000 sets
  variance across contexts=0.1839; across targets=0.9186; target/context=5.00x
  sets with higher across-target variance: 97.1%
  variance parts: context baseline 3.0%, target baseline 80.6%, context-target interaction 16.5%
  matched-pair interaction lift=+0.2727; positive in 72.4% of sets
  median absolute matched-vs-other margin: fixed target 0.1497, fixed context 0.6719
  pairwise accuracy (set mean): context 64.9%, target 58.5% (chance 50%)
  column-centered target accuracy: pairwise 68.1%, top target 52.4% (diagnostic only)
  same top target in every context: 63.1% of sets; correct top target 36.4% (chance 25.4%)
  target index 0 ranked top: 29.0% of contexts (uniform index baseline 25.4%); top ties 0.3%
Benchmarks with higher across-target variance: 30/30

icl: 500 sets
  variance across contexts=0.0037; across targets=0.1851; target/context=49.41x
  sets with higher across-target variance: 99.6%
  variance parts: context baseline 1.2%, target baseline 98.0%, context-target interaction 0.8%
  matched-pair interaction lift=+0.0180; positive in 82.6% of sets
  median absolute matched-vs-other margin: fixed target 0.0545, fixed context 0.4397
  pairwise accuracy (set mean): context 60.7%, target 51.1% (chance 50%)
  column-centered target accuracy: pairwise 66.8%, top target 43.8% (diagnostic only)
  same top target in every context: 86.8% of sets; correct top target 25.6% (chance 25.0%)
  target index 0 ranked top: 38.1% of contexts (uniform index baseline 25.0%); top ties 0.0%

knowledge: 500 sets
  variance across contexts=0.2057; across targets=0.8035; target/context=3.91x
  sets with higher across-target variance: 95.8%
  variance parts: context baseline 6.3%, target baseline 76.0%, context-target interaction 17.7%
  matched-pair interaction lift=+0.4006; positive in 90.4% of sets
  median absolute matched-vs-other margin: fixed target 0.4136, fixed context 0.9514
  pairwise accuracy (set mean): context 75.9%, target 64.3% (chance 50%)
  column-centered target accuracy: pairwise 80.5%, top target 65.5% (diagnostic only)
  same top target in every context: 53.2% of sets; correct top target 38.5% (chance 25.0%)
  target index 0 ranked top: 25.9% of contexts (uniform index baseline 25.0%); top ties 0.0%

language-manipulation: 500 sets
  variance across contexts=0.8265; across targets=2.1794; target/context=2.64x
  sets with higher across-target variance: 97.4%
  variance parts: context baseline 3.6%, target baseline 63.4%, context-target interaction 32.9%
  matched-pair interaction lift=+1.2049; positive in 100.0% of sets
  median absolute matched-vs-other margin: fixed target 1.4139, fixed context 1.4766
  pairwise accuracy (set mean): context 98.3%, target 82.1% (chance 50%)
  column-centered target accuracy: pairwise 99.5%, top target 98.6% (diagnostic only)
  same top target in every context: 13.2% of sets; correct top target 66.1% (chance 25.0%)
  target index 0 ranked top: 25.0% of contexts (uniform index baseline 25.0%); top ties 0.1%

math: 500 sets
  variance across contexts=0.0302; across targets=1.4411; target/context=47.78x
  sets with higher across-target variance: 95.0%
  variance parts: context baseline 0.8%, target baseline 97.9%, context-target interaction 1.3%
  matched-pair interaction lift=+0.0932; positive in 70.8% of sets
  median absolute matched-vs-other margin: fixed target 0.1697, fixed context 0.4844
  pairwise accuracy (set mean): context 65.9%, target 58.3% (chance 50%)
  column-centered target accuracy: pairwise 70.0%, top target 59.1% (diagnostic only)
  same top target in every context: 59.2% of sets; correct top target 41.6% (chance 27.8%)
  target index 0 ranked top: 32.0% of contexts (uniform index baseline 27.8%); top ties 1.2%

math-languages: 500 sets
  variance across contexts=0.0069; across targets=0.5445; target/context=79.27x
  sets with higher across-target variance: 95.4%
  variance parts: context baseline 1.0%, target baseline 98.8%, context-target interaction 0.3%
  matched-pair interaction lift=+0.0006; positive in 52.6% of sets
  median absolute matched-vs-other margin: fixed target 0.0377, fixed context 0.5352
  pairwise accuracy (set mean): context 50.2%, target 50.4% (chance 50%)
  column-centered target accuracy: pairwise 53.0%, top target 28.9% (diagnostic only)
  same top target in every context: 90.0% of sets; correct top target 25.3% (chance 25.0%)
  target index 0 ranked top: 36.2% of contexts (uniform index baseline 25.0%); top ties 0.2%

reasoning: 500 sets
  variance across contexts=0.0302; across targets=0.3581; target/context=11.87x
  sets with higher across-target variance: 99.2%
  variance parts: context baseline 3.7%, target baseline 91.9%, context-target interaction 4.4%
  matched-pair interaction lift=-0.0810; positive in 38.2% of sets
  median absolute matched-vs-other margin: fixed target 0.1424, fixed context 0.6454
  pairwise accuracy (set mean): context 38.4%, target 44.8% (chance 50%)
  column-centered target accuracy: pairwise 39.0%, top target 19.2% (diagnostic only)
  same top target in every context: 76.2% of sets; correct top target 21.7% (chance 25.0%)
  target index 0 ranked top: 17.0% of contexts (uniform index baseline 25.0%); top ties 0.4%

Interpretation: across-target minus across-context variance equals target-baseline minus context-baseline variance. A large ratio shows stable target preferences, not that target rankings are random.
```
