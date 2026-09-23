# Synthetic benchmark configuration specification

Version 1 describes how to generate related `(context, target)` pairs. It does
not configure how a model is scored. An evaluator is expected to score every
target against every context in the same contrast set:

```text
S[i,j] = log P(target[j] | context[i])
```

The configuration is one JSON object. All sources are embedded in that object,
so a benchmark is self-contained.

## Top-level object

```json
{
  "version": 1,
  "id": "object_counting",
  "language": "it",
  "generation": {
    "sets": 1000,
    "seed": 42,
    "max_attempts_per_set": 100
  },
  "sources": {},
  "variables": {},
  "derived": {},
  "constraints": [],
  "contrast_set": {},
  "templates": []
}
```

Required fields are `version`, `id`, `variables`, `contrast_set`, and
`templates`. `version` must be `1`.

The optional `generation` object has these fields:

- `sets`: number of contrast sets to generate; default `1`.
- `seed`: random seed; default `0`.
- `max_attempts_per_set`: number of times an invalid set may be resampled;
  default `100`.

`language` is optional metadata and does not affect generation.

## Sources

`sources` maps names to nonempty lists. Values may be scalars:

```json
{
  "sources": {
    "fruits": ["mele", "pere", "arance"]
  }
}
```

or records:

```json
{
  "sources": {
    "country_capitals": [
      {"country": "Italia", "capital": "Roma"},
      {"country": "Francia", "capital": "Parigi"}
    ]
  }
}
```

## Variables

`variables` defines the values sampled for one base world. Version 1 has three
variable types.

### Constant

```json
{"type": "constant", "value": "mele"}
```

### Choice

Samples one value uniformly from a named source:

```json
{"type": "choice", "source": "fruits"}
```

### Integer

Samples uniformly from an inclusive integer range:

```json
{"type": "integer", "min": 2, "max": 8, "step": 1}
```

`step` is optional and defaults to `1`.

### Sample

Samples several values from a source and returns them as a list:

```json
{"type": "sample", "source": "names", "count": 4, "replace": false}
```

`replace` is optional and defaults to `false`. Without replacement, the source
must contain at least `count` distinct values.

## References

Strings starting with `$` are references. A reference may select a complete
value or a field within a record:

```text
$total
$fact.country
$fact.capital
```

All other strings are literals.

## Derived values

`derived` maps names to expression trees. Derived entries are evaluated in
declaration order, so an entry may refer to variables and to derived values
declared above it.

```json
{
  "derived": {
    "second_count": {
      "op": "subtract",
      "args": ["$total", "$first_count"]
    }
  }
}
```

Arithmetic and sequence operators are `add`, `subtract`, `multiply`,
`minimum`, `maximum`, `length`, `integer_range`, `repeat`, `interleave`,
`concatenate`, `pair_strings`, `count_equal`, `item_at`, `remove_at`, `slice`,
and `pluck`. Expressions may be nested. `subtract` takes exactly two arguments
and `length` takes exactly one.
`integer_range` takes an integer start and end and returns their inclusive
ascending range; it is limited to 10,001 values. `repeat` creates a list from a
value and count, `interleave` combines lists in round-robin order,
`concatenate` joins lists, `pair_strings` joins corresponding values from two
equal-length lists using a separator, `count_equal` counts values equal to a
given value, and `item_at` selects a zero-based list position. `remove_at`
returns a list without one indexed value, `slice` selects a half-open list
range, and `pluck` extracts one field from every record in a list.

## Constraints

Each constraint is an expression that must evaluate to `true`:

```json
{
  "constraints": [
    {
      "op": "greater_than",
      "args": ["$second_count", 0]
    }
  ]
}
```

Boolean operators are `equal`, `not_equal`, `less_than`, `less_or_equal`,
`greater_than`, `greater_or_equal`, `and`, `or`, and `not`. Comparisons take
two arguments, `not` takes one, and `and` and `or` take one or more.

If any member fails a constraint, the entire contrast set is discarded and
resampled.

## Contrast sets and mutations

`contrast_set.size` must be at least two. A base world is copied `size` times,
then mutations assign related values to the copies. Variables not named by a
mutation remain identical across the set.

```json
{
  "contrast_set": {
    "size": 4,
    "mutations": [],
    "unique_on": []
  }
}
```

Mutations may only target variables, not derived values. After all mutations,
derived values are recomputed independently for every member.

### Offset

Adds each offset to the sampled base value:

```json
{
  "variable": "total",
  "operation": "offset",
  "values": [0, 1, 2, 3]
}
```

The number of offsets must equal `contrast_set.size`.

### Replace

Assigns explicit values to the members:

```json
{
  "variable": "direction",
  "operation": "replace",
  "values": ["nord", "sud", "est", "ovest"]
}
```

The number of values must equal `contrast_set.size`.

### Distinct resampling

Samples distinct entries from the source of a `choice` variable:

```json
{
  "variable": "fact",
  "operation": "resample_distinct"
}
```

The source must contain at least `contrast_set.size` distinct values.

When several mutations are present, their results are zipped by member index;
no Cartesian product is produced. A variable may be mutated at most once.

`unique_on` is an optional list of references. Each referenced value must be
different in every member:

```json
{"unique_on": ["$total"]}
```

## Templates

`templates` is a nonempty list of paired context and target strings:

```json
{
  "templates": [
    {
      "context": "Ci sono {first_count|it_number} {object} e ne vengono aggiunte altre {second_count|it_number}.",
      "target": " In totale ci sono {total|it_number} {object}."
    }
  ]
}
```

One template pair is sampled per contrast set and used for every member in
that set. A placeholder may contain a value, a record field, or a formatter:

```text
{variable}
{record.field}
{variable|formatter}
```

Version 1 defines two formatters:

- `str`: the default textual representation.
- `it_number`: an Italian representation of an integer from -99 through 99.
- `comma_list`: joins the values in a list with a comma and a space.

Whitespace belongs to the template. In particular, a target may intentionally
start with a space when it continues its context.

## Generation algorithm

For every requested contrast set, the engine performs these steps:

```text
sample one template pair
sample one base world
copy the base world N times
apply mutations to the N members
recompute derived values for every member
validate constraints and uniqueness
render contexts and targets
```

If validation fails, the whole attempt, including its template, is resampled.
Generation fails after `max_attempts_per_set` unsuccessful attempts.

## Complete example

```json
{
  "version": 1,
  "id": "object_counting",
  "language": "it",
  "generation": {
    "sets": 3,
    "seed": 42,
    "max_attempts_per_set": 100
  },
  "sources": {
    "fruits": ["mele", "pere", "arance"]
  },
  "variables": {
    "object": {"type": "choice", "source": "fruits"},
    "total": {"type": "integer", "min": 4, "max": 6},
    "first_count": {"type": "integer", "min": 2, "max": 3}
  },
  "derived": {
    "second_count": {
      "op": "subtract",
      "args": ["$total", "$first_count"]
    }
  },
  "constraints": [
    {
      "op": "greater_than",
      "args": ["$second_count", 1]
    }
  ],
  "contrast_set": {
    "size": 4,
    "mutations": [
      {
        "variable": "total",
        "operation": "offset",
        "values": [0, 1, 2, 3]
      }
    ],
    "unique_on": ["$total"]
  },
  "templates": [
    {
      "context": "Ci sono {first_count|it_number} {object} e ne vengono aggiunte altre {second_count|it_number}.",
      "target": " In totale ci sono {total|it_number} {object}."
    }
  ]
}
```

## Reference generator

Generate all contrast sets and print one context-target pair as a preview with:

```bash
python src/evaluation/generate_synthetic_benchmark.py \
  --src-config benchmark.json \
  --output generated.json
```

`--src-config` selects the generation configuration and `--output` selects the
generated JSON path. The output file contains all sets in this form:

```json
{
  "version": 1,
  "id": "object_counting",
  "language": "it",
  "sets": [
    {
      "members": [
        {
          "context": "Ci sono due mele e ne vengono aggiunte altre due.",
          "target": " In totale ci sono quattro mele."
        }
      ]
    }
  ]
}
```

Every generated member is written to the file. Standard output contains only
the first member, as a short human-readable preview:

```text
CONTEXT: Ci sono due mele e ne vengono aggiunte altre due.
TARGET:  In totale ci sono quattro mele.
```

## Reference scorer

Score a generated benchmark with:

```bash
python src/evaluation/score_synthetic_benchmark.py \
  --checkpoint models/model.pt \
  --tokenizer src/tokenizer/production_16k/tokenizer.model \
  --benchmark generated.json \
  --output scores.json
```

The scorer uses teacher forcing to compute the full matrix
`S[i,j] = log P(target[j] | context[i])`. It sums the conditional
log-probabilities of every target token and does not include EOS. The generated
report contains the score matrix, mean-token score matrix, individual token
log-probabilities, target tokenization, per-set metrics, and aggregate metrics.

Macro pairwise accuracy is the primary metric. The report also includes micro
pairwise accuracy, context-retrieval accuracy, mean reciprocal rank,
worst-distractor margins, and perfect-set accuracy. Scoring is fixed by the
evaluator and is not a field in the generation configuration.
