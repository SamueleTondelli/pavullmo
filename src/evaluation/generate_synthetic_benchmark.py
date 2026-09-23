"""Generate a benchmark JSON file and print one context-target preview.

Usage:
    python src/evaluation/generate_synthetic_benchmark.py \
        --src-config CONFIG --output OUTPUT
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any


PLACEHOLDER = re.compile(
    r"\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"(?:\|([A-Za-z_][A-Za-z0-9_]*))?\}"
)


class ConfigError(ValueError):
    """A benchmark configuration is invalid or cannot generate a valid set."""


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list")
    return value


def require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ConfigError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{label} must be at least {minimum}")
    return value


def resolve_path(values: dict[str, Any], path: str, label: str) -> Any:
    parts = path.split(".")
    if not parts or not parts[0]:
        raise ConfigError(f"{label} contains an empty reference")
    try:
        value: Any = values[parts[0]]
        for part in parts[1:]:
            if not isinstance(value, dict):
                raise ConfigError(
                    f"{label} cannot select {part!r} from a non-record value"
                )
            value = value[part]
        return value
    except KeyError as error:
        raise ConfigError(f"{label} references unknown value {path!r}") from error


def resolve_reference(values: dict[str, Any], reference: Any, label: str) -> Any:
    if not isinstance(reference, str) or not reference.startswith("$"):
        raise ConfigError(f"{label} must be a reference beginning with '$'")
    return resolve_path(values, reference[1:], label)


def evaluate(expression: Any, values: dict[str, Any], label: str) -> Any:
    if isinstance(expression, str) and expression.startswith("$"):
        return resolve_reference(values, expression, label)
    if not isinstance(expression, dict):
        return expression

    operation = expression.get("op")
    if not isinstance(operation, str):
        raise ConfigError(f"{label} expression must contain a string 'op'")
    arguments = require_list(expression.get("args"), f"{label}.args")
    args = [evaluate(argument, values, label) for argument in arguments]

    binary_operations = {
        "subtract": lambda a, b: a - b,
        "equal": lambda a, b: a == b,
        "not_equal": lambda a, b: a != b,
        "less_than": lambda a, b: a < b,
        "less_or_equal": lambda a, b: a <= b,
        "greater_than": lambda a, b: a > b,
        "greater_or_equal": lambda a, b: a >= b,
    }
    if operation in binary_operations:
        if len(args) != 2:
            raise ConfigError(f"{label}: {operation} requires exactly two arguments")
        return binary_operations[operation](args[0], args[1])
    if operation == "length":
        if len(args) != 1:
            raise ConfigError(f"{label}: length requires exactly one argument")
        return len(args[0])
    if operation == "integer_range":
        if len(args) != 2:
            raise ConfigError(
                f"{label}: integer_range requires exactly two arguments"
            )
        start, end = args
        if type(start) is not int or type(end) is not int:
            raise ConfigError(f"{label}: integer_range arguments must be integers")
        if start > end:
            raise ConfigError(
                f"{label}: integer_range start must not exceed its end"
            )
        if end - start > 10_000:
            raise ConfigError(
                f"{label}: integer_range may contain at most 10,001 values"
            )
        return list(range(start, end + 1))
    if operation == "repeat":
        if len(args) != 2:
            raise ConfigError(f"{label}: repeat requires exactly two arguments")
        value, count = args
        if type(count) is not int or not 0 <= count <= 10_000:
            raise ConfigError(
                f"{label}: repeat count must be an integer from 0 through 10,000"
            )
        return [copy.deepcopy(value) for _ in range(count)]
    if operation == "interleave":
        if not args:
            raise ConfigError(f"{label}: interleave requires at least one argument")
        if any(not isinstance(value, list) for value in args):
            raise ConfigError(f"{label}: interleave arguments must be lists")
        result: list[Any] = []
        for index in range(max(len(value) for value in args)):
            for value in args:
                if index < len(value):
                    result.append(copy.deepcopy(value[index]))
        return result
    if operation == "concatenate":
        if not args or any(not isinstance(value, list) for value in args):
            raise ConfigError(
                f"{label}: concatenate requires one or more lists"
            )
        return [copy.deepcopy(item) for value in args for item in value]
    if operation == "pair_strings":
        if (
            len(args) != 3
            or not isinstance(args[0], list)
            or not isinstance(args[1], str)
            or not isinstance(args[2], list)
        ):
            raise ConfigError(
                f"{label}: pair_strings requires a list, separator, and list"
            )
        left, separator, right = args
        if len(left) != len(right):
            raise ConfigError(f"{label}: pair_strings lists must have equal length")
        return [
            stringify(left_value) + separator + stringify(right_value)
            for left_value, right_value in zip(left, right, strict=True)
        ]
    if operation == "count_equal":
        if len(args) != 2 or not isinstance(args[0], list):
            raise ConfigError(
                f"{label}: count_equal requires a list and a comparison value"
            )
        return sum(item == args[1] for item in args[0])
    if operation == "item_at":
        if len(args) != 2 or not isinstance(args[0], list):
            raise ConfigError(f"{label}: item_at requires a list and an index")
        sequence, index = args
        if type(index) is not int or not 0 <= index < len(sequence):
            raise ConfigError(f"{label}: item_at index is outside the list")
        return copy.deepcopy(sequence[index])
    if operation == "remove_at":
        if len(args) != 2 or not isinstance(args[0], list):
            raise ConfigError(f"{label}: remove_at requires a list and an index")
        sequence, index = args
        if type(index) is not int or not 0 <= index < len(sequence):
            raise ConfigError(f"{label}: remove_at index is outside the list")
        return copy.deepcopy(sequence[:index] + sequence[index + 1 :])
    if operation == "slice":
        if len(args) != 3 or not isinstance(args[0], list):
            raise ConfigError(
                f"{label}: slice requires a list, start index, and end index"
            )
        sequence, start, end = args
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= len(sequence)
        ):
            raise ConfigError(f"{label}: slice indices are outside the list")
        return copy.deepcopy(sequence[start:end])
    if operation == "pluck":
        if (
            len(args) != 2
            or not isinstance(args[0], list)
            or not isinstance(args[1], str)
        ):
            raise ConfigError(f"{label}: pluck requires a list and a field name")
        records, field = args
        result = []
        for record in records:
            if not isinstance(record, dict):
                raise ConfigError(f"{label}: pluck values must be records")
            result.append(copy.deepcopy(resolve_path(record, field, label)))
        return result
    if operation == "not":
        if len(args) != 1:
            raise ConfigError(f"{label}: not requires exactly one argument")
        return not args[0]
    if operation in {"add", "multiply", "minimum", "maximum", "and", "or"}:
        if not args:
            raise ConfigError(f"{label}: {operation} requires at least one argument")
        if operation == "add":
            return sum(args)
        if operation == "multiply":
            return reduce(mul, args)
        if operation == "minimum":
            return min(args)
        if operation == "maximum":
            return max(args)
        if operation == "and":
            return all(args)
        return any(args)
    raise ConfigError(f"{label} uses unknown operation {operation!r}")


def sample_variable(
    name: str,
    specification: Any,
    sources: dict[str, Any],
    rng: random.Random,
) -> Any:
    spec = require_object(specification, f"variables.{name}")
    variable_type = spec.get("type")
    if variable_type == "constant":
        if "value" not in spec:
            raise ConfigError(f"variables.{name} constant is missing 'value'")
        return copy.deepcopy(spec["value"])
    if variable_type == "choice":
        source_name = spec.get("source")
        if not isinstance(source_name, str) or source_name not in sources:
            raise ConfigError(f"variables.{name}.source must name an existing source")
        source = require_list(sources[source_name], f"sources.{source_name}")
        if not source:
            raise ConfigError(f"sources.{source_name} must not be empty")
        return copy.deepcopy(rng.choice(source))
    if variable_type == "integer":
        minimum = require_int(spec.get("min"), f"variables.{name}.min")
        maximum = require_int(spec.get("max"), f"variables.{name}.max")
        step = require_int(spec.get("step", 1), f"variables.{name}.step", minimum=1)
        if minimum > maximum:
            raise ConfigError(f"variables.{name}.min must not exceed max")
        return rng.choice(range(minimum, maximum + 1, step))
    if variable_type == "sample":
        source_name = spec.get("source")
        if not isinstance(source_name, str) or source_name not in sources:
            raise ConfigError(
                f"variables.{name}.source must name an existing source"
            )
        count = require_int(
            spec.get("count"), f"variables.{name}.count", minimum=1
        )
        replace = spec.get("replace", False)
        if type(replace) is not bool:
            raise ConfigError(f"variables.{name}.replace must be a boolean")
        source = require_list(sources[source_name], f"sources.{source_name}")
        if replace:
            return [copy.deepcopy(rng.choice(source)) for _ in range(count)]
        unique_source = distinct_values(source)
        if len(unique_source) < count:
            raise ConfigError(
                f"sources.{source_name} needs at least {count} distinct values"
            )
        return copy.deepcopy(rng.sample(unique_source, count))
    raise ConfigError(f"variables.{name} has unknown type {variable_type!r}")


def distinct_values(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def apply_mutations(
    members: list[dict[str, Any]],
    base: dict[str, Any],
    mutations: list[Any],
    variable_specs: dict[str, Any],
    sources: dict[str, Any],
    rng: random.Random,
) -> None:
    size = len(members)
    mutated: set[str] = set()
    for index, value in enumerate(mutations):
        mutation = require_object(value, f"contrast_set.mutations[{index}]")
        variable = mutation.get("variable")
        if not isinstance(variable, str) or variable not in variable_specs:
            raise ConfigError(
                f"contrast_set.mutations[{index}].variable must name a variable"
            )
        if variable in mutated:
            raise ConfigError(f"variable {variable!r} is mutated more than once")
        mutated.add(variable)
        operation = mutation.get("operation")

        if operation in {"offset", "replace"}:
            mutation_values = require_list(
                mutation.get("values"),
                f"contrast_set.mutations[{index}].values",
            )
            if len(mutation_values) != size:
                raise ConfigError(
                    f"contrast_set.mutations[{index}].values must contain "
                    f"exactly {size} entries"
                )
            if operation == "offset":
                base_value = base[variable]
                if isinstance(base_value, bool) or not isinstance(
                    base_value, (int, float)
                ):
                    raise ConfigError(f"offset mutation requires numeric {variable!r}")
                for member, offset in zip(members, mutation_values, strict=True):
                    if isinstance(offset, bool) or not isinstance(offset, (int, float)):
                        raise ConfigError("offset values must be numbers")
                    member[variable] = base_value + offset
            else:
                for member, replacement in zip(members, mutation_values, strict=True):
                    member[variable] = copy.deepcopy(replacement)
            continue

        if operation == "resample_distinct":
            variable_spec = require_object(
                variable_specs[variable], f"variables.{variable}"
            )
            if variable_spec.get("type") != "choice":
                raise ConfigError(
                    "resample_distinct requires a variable of type 'choice'"
                )
            source_name = variable_spec.get("source")
            if not isinstance(source_name, str) or source_name not in sources:
                raise ConfigError(f"variables.{variable}.source is invalid")
            source = distinct_values(
                require_list(sources[source_name], f"sources.{source_name}")
            )
            if len(source) < size:
                raise ConfigError(
                    f"source {source_name!r} needs at least {size} distinct values"
                )
            sampled = rng.sample(source, size)
            for member, replacement in zip(members, sampled, strict=True):
                member[variable] = copy.deepcopy(replacement)
            continue

        raise ConfigError(
            f"contrast_set.mutations[{index}] has unknown operation {operation!r}"
        )


def italian_number(value: Any) -> str:
    if type(value) is not int or not -99 <= value <= 99:
        raise ConfigError("it_number requires an integer from -99 through 99")
    if value < 0:
        return "meno " + italian_number(-value)
    units = [
        "zero",
        "uno",
        "due",
        "tre",
        "quattro",
        "cinque",
        "sei",
        "sette",
        "otto",
        "nove",
        "dieci",
        "undici",
        "dodici",
        "tredici",
        "quattordici",
        "quindici",
        "sedici",
        "diciassette",
        "diciotto",
        "diciannove",
    ]
    if value < 20:
        return units[value]
    tens = [
        "",
        "",
        "venti",
        "trenta",
        "quaranta",
        "cinquanta",
        "sessanta",
        "settanta",
        "ottanta",
        "novanta",
    ]
    tens_word = tens[value // 10]
    unit = value % 10
    if unit == 0:
        return tens_word
    if unit in {1, 8}:
        tens_word = tens_word[:-1]
    if unit == 3:
        return tens_word + "tré"
    return tens_word + units[unit]


def stringify(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def render(template: str, values: dict[str, Any], label: str) -> str:
    def replace(match: re.Match[str]) -> str:
        path, formatter = match.groups()
        value = resolve_path(values, path, label)
        if formatter in {None, "str"}:
            return stringify(value)
        if formatter == "it_number":
            return italian_number(value)
        if formatter == "comma_list":
            if not isinstance(value, list):
                raise ConfigError(f"{label}: comma_list requires a list")
            return ", ".join(stringify(item) for item in value)
        raise ConfigError(f"{label} uses unknown formatter {formatter!r}")

    rendered = PLACEHOLDER.sub(replace, template)
    if "{" in rendered or "}" in rendered:
        raise ConfigError(f"{label} contains an invalid placeholder")
    return rendered


def generate(config: Any) -> list[list[tuple[str, str]]]:
    root = require_object(config, "configuration")
    if root.get("version") != 1:
        raise ConfigError("version must be 1")
    if not isinstance(root.get("id"), str) or not root["id"]:
        raise ConfigError("id must be a nonempty string")

    generation = require_object(root.get("generation", {}), "generation")
    set_count = require_int(generation.get("sets", 1), "generation.sets", minimum=1)
    seed = require_int(generation.get("seed", 0), "generation.seed")
    max_attempts = require_int(
        generation.get("max_attempts_per_set", 100),
        "generation.max_attempts_per_set",
        minimum=1,
    )
    rng = random.Random(seed)

    sources = require_object(root.get("sources", {}), "sources")
    for source_name, source in sources.items():
        if not isinstance(source_name, str) or not source_name:
            raise ConfigError("source names must be nonempty strings")
        if not require_list(source, f"sources.{source_name}"):
            raise ConfigError(f"sources.{source_name} must not be empty")

    variables = require_object(root.get("variables"), "variables")
    derived = require_object(root.get("derived", {}), "derived")
    duplicate_names = set(variables) & set(derived)
    if duplicate_names:
        raise ConfigError(
            f"variable and derived names overlap: {sorted(duplicate_names)!r}"
        )
    constraints = require_list(root.get("constraints", []), "constraints")

    contrast = require_object(root.get("contrast_set"), "contrast_set")
    size = require_int(contrast.get("size"), "contrast_set.size", minimum=2)
    mutations = require_list(contrast.get("mutations", []), "contrast_set.mutations")
    if not mutations:
        raise ConfigError("contrast_set.mutations must not be empty")
    unique_on = require_list(contrast.get("unique_on", []), "contrast_set.unique_on")

    templates = require_list(root.get("templates"), "templates")
    if not templates:
        raise ConfigError("templates must not be empty")
    parsed_templates: list[tuple[str, str]] = []
    for index, value in enumerate(templates):
        template = require_object(value, f"templates[{index}]")
        context = template.get("context")
        target = template.get("target")
        if not isinstance(context, str) or not isinstance(target, str):
            raise ConfigError(
                f"templates[{index}] must contain string context and target"
            )
        parsed_templates.append((context, target))

    generated: list[list[tuple[str, str]]] = []
    for set_index in range(set_count):
        for _attempt in range(max_attempts):
            context_template, target_template = rng.choice(parsed_templates)
            base = {
                name: sample_variable(name, spec, sources, rng)
                for name, spec in variables.items()
            }
            members = [copy.deepcopy(base) for _ in range(size)]
            apply_mutations(members, base, mutations, variables, sources, rng)

            valid = True
            for member_index, member in enumerate(members):
                for name, expression in derived.items():
                    member[name] = evaluate(
                        expression,
                        member,
                        f"derived.{name} for member {member_index}",
                    )
                for constraint_index, constraint in enumerate(constraints):
                    result = evaluate(
                        constraint,
                        member,
                        f"constraints[{constraint_index}] for member {member_index}",
                    )
                    if type(result) is not bool:
                        raise ConfigError(
                            f"constraints[{constraint_index}] must return a boolean"
                        )
                    if not result:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                for reference in unique_on:
                    resolved = [
                        resolve_reference(member, reference, "contrast_set.unique_on")
                        for member in members
                    ]
                    if len(distinct_values(resolved)) != size:
                        valid = False
                        break
            if not valid:
                continue

            rendered = [
                (
                    render(
                        context_template,
                        member,
                        f"context for set {set_index}, member {member_index}",
                    ),
                    render(
                        target_template,
                        member,
                        f"target for set {set_index}, member {member_index}",
                    ),
                )
                for member_index, member in enumerate(members)
            ]
            generated.append(rendered)
            break
        else:
            raise ConfigError(
                f"could not generate valid contrast set {set_index} after "
                f"{max_attempts} attempts"
            )
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-config",
        type=Path,
        required=True,
        help="path to the benchmark configuration JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path for the generated benchmark JSON",
    )
    return parser.parse_args()


def output_document(
    config: dict[str, Any], sets: list[list[tuple[str, str]]]
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "version": 1,
        "id": config["id"],
        "sets": [
            {
                "members": [
                    {"context": context, "target": target}
                    for context, target in members
                ]
            }
            for members in sets
        ],
    }
    if "language" in config:
        document["language"] = config["language"]
    return document


def main() -> None:
    args = parse_args()
    try:
        if args.src_config.resolve() == args.output.resolve():
            raise ConfigError("config and output paths must be different")
        with args.src_config.open(encoding="utf-8") as file:
            config = json.load(file)
        sets = generate(config)
        root = require_object(config, "configuration")
        document = output_document(root, sets)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(document, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except (OSError, json.JSONDecodeError, ConfigError, TypeError) as error:
        raise SystemExit(f"error: {error}") from error

    for context, target in sets[0]:
        print(f"CONTEXT: {context}")
        print(f"TARGET: {target}")


if __name__ == "__main__":
    main()
