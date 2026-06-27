from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import random
import re
from typing import Sequence


@dataclass(frozen=True)
class CountdownCheck:
    correct: bool
    valid_equations: tuple[bool, ...]
    prediction: str


class CountdownTokenizer:
    """Atomic integer tokenizer for Countdown arithmetic traces."""

    task = "countdown"

    def __init__(self, *, max_intermediate: int = 10_000):
        if max_intermediate < 1:
            raise ValueError("max_intermediate must be positive")
        self.max_intermediate = int(max_intermediate)
        symbol_base = self.max_intermediate + 1
        self.pipe_id = symbol_base
        self.mul_id = symbol_base + 1
        self.div_id = symbol_base + 2
        self.plus_id = symbol_base + 3
        self.minus_id = symbol_base + 4
        self.equals_id = symbol_base + 5
        self.comma_id = symbol_base + 6
        self.eos_id = symbol_base + 7
        self.pad_id = symbol_base + 8
        self._encoder = {
            "|": self.pipe_id,
            "*": self.mul_id,
            "/": self.div_id,
            "+": self.plus_id,
            "-": self.minus_id,
            "=": self.equals_id,
            ",": self.comma_id,
        }
        self._decoder = {
            self.pipe_id: "|",
            self.mul_id: "*",
            self.div_id: "/",
            self.plus_id: "+",
            self.minus_id: "-",
            self.equals_id: "=",
            self.comma_id: ",",
        }

    @property
    def vocab_size(self) -> int:
        return self.pad_id + 1

    def _number_id(self, text: str) -> int:
        value = int(text)
        if value < 0 or value > self.max_intermediate:
            raise ValueError(
                f"number {value} is outside tokenizer range 0..{self.max_intermediate}"
            )
        return value

    def encode(
        self,
        text: str,
        *,
        num_pause_tokens: int = 0,
        add_eos: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        index = 0
        seen_pipe = False
        while index < len(text):
            char = text[index]
            if char.isspace():
                index += 1
                continue
            if char == "," and not seen_pipe:
                index += 1
                continue
            if char.isdigit():
                start = index
                while index < len(text) and text[index].isdigit():
                    index += 1
                ids.append(self._number_id(text[start:index]))
                continue
            if char == "|":
                seen_pipe = True
                ids.extend([self.pipe_id] * int(num_pause_tokens))
                index += 1
                continue
            if char not in self._encoder:
                raise ValueError(f"unsupported Countdown character: {char!r}")
            ids.append(self._encoder[char])
            index += 1
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def tokenize(
        self,
        text: str,
        *,
        num_pause_tokens: int,
    ) -> tuple[list[int], int]:
        line = text.strip()
        if "|" not in line:
            raise ValueError("Countdown examples must contain a prompt/solution pipe")
        prompt = line.split("|", 1)[0] + "|"
        prompt_ids = self.encode(prompt, num_pause_tokens=num_pause_tokens)
        ids = self.encode(line, num_pause_tokens=num_pause_tokens, add_eos=True)
        return ids, len(prompt_ids)

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        pieces = []
        for raw_id in ids:
            token_id = int(raw_id)
            if 0 <= token_id <= self.max_intermediate:
                pieces.append(str(token_id))
            elif token_id in (self.eos_id, self.pad_id):
                if skip_special_tokens:
                    continue
                pieces.append("<eos>" if token_id == self.eos_id else "<pad>")
            else:
                pieces.append(self._decoder.get(token_id, ""))
        return "".join(pieces)

    def number_tokens(self, ids: Sequence[int]) -> list[int]:
        return [
            int(token_id)
            for token_id in ids
            if 0 <= int(token_id) <= self.max_intermediate
        ]


def countdown_solution_token_count(num_equations: int) -> int:
    if num_equations < 1:
        raise ValueError("num_equations must be positive")
    return 5 * num_equations + (num_equations - 1)


def _combine_numbers(
    left: int,
    right: int,
    *,
    max_intermediate: int,
) -> list[tuple[int, str]]:
    possible: list[tuple[int, str]] = []
    if left + right <= max_intermediate:
        possible.append((left + right, f"{left}+{right}={left + right}"))
    if left * right <= max_intermediate:
        possible.append((left * right, f"{left}*{right}={left * right}"))
    if left <= right:
        possible.append((right - left, f"{right}-{left}={right - left}"))
        if left != 0 and right % left == 0:
            possible.append((right // left, f"{right}/{left}={right // left}"))
    else:
        possible.append((left - right, f"{left}-{right}={left - right}"))
        if right != 0 and left % right == 0:
            possible.append((left // right, f"{left}/{right}={left // right}"))
    return possible


def _search_solution(
    target: int,
    numbers: list[int],
    *,
    max_intermediate: int,
    operations: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    if len(numbers) == 1:
        return operations if numbers[0] == target else None
    for left_index, right_index in itertools.combinations(range(len(numbers)), 2):
        left = numbers[left_index]
        right = numbers[right_index]
        remaining = [
            value
            for index, value in enumerate(numbers)
            if index not in {left_index, right_index}
        ]
        for result, operation in _combine_numbers(
            left,
            right,
            max_intermediate=max_intermediate,
        ):
            found = _search_solution(
                target,
                [*remaining, result],
                max_intermediate=max_intermediate,
                operations=(*operations, operation),
            )
            if found is not None:
                return found
    return None


def target_splits(
    *,
    min_target: int,
    max_target: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if min_target >= max_target:
        raise ValueError("min_target must be less than max_target")
    targets = list(range(min_target, max_target + 1))
    rng = random.Random(seed)
    rng.shuffle(targets)
    heldout_count = max(1, len(targets) // 10)
    heldout = sorted(targets[:heldout_count])
    train = sorted(targets[heldout_count:])
    return train, heldout


def generate_countdown_example(
    rng: random.Random,
    *,
    target_pool: Sequence[int],
    input_numbers: int,
    max_target: int,
    max_intermediate: int,
) -> str:
    if input_numbers < 2:
        raise ValueError("input_numbers must be at least 2")
    if not target_pool:
        raise ValueError("target_pool must not be empty")
    for _ in range(20_000):
        target = int(rng.choice(target_pool))
        numbers = [rng.randint(1, max_target - 1) for _ in range(input_numbers)]
        solution = _search_solution(
            target,
            numbers,
            max_intermediate=max_intermediate,
        )
        if solution is not None:
            prefix = ",".join(str(number) for number in numbers)
            return f"{prefix},{target}|" + ",".join(solution)
    raise RuntimeError("failed to generate a solvable Countdown example")


_EQUATION_RE = re.compile(r"^(\d+)([+\-*/])(\d+)=(\d+)$")


def _evaluate_binary(left: int, operator: str, right: int) -> int | None:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0 or left % right != 0:
            return None
        return left // right
    raise ValueError(f"unknown operator: {operator}")


def check_countdown_solution(
    *,
    input_numbers: Sequence[int],
    target: int,
    prediction: str,
    num_equations: int,
) -> CountdownCheck:
    equations = prediction.split(",") if prediction else []
    available = Counter(int(number) for number in input_numbers)
    valid: list[bool] = []
    last_result: int | None = None
    for index in range(num_equations):
        if index >= len(equations):
            valid.append(False)
            continue
        match = _EQUATION_RE.match(equations[index])
        if match is None:
            valid.append(False)
            continue
        left = int(match.group(1))
        operator = match.group(2)
        right = int(match.group(3))
        result = int(match.group(4))
        used = Counter((left, right))
        if any(available[number] < count for number, count in used.items()):
            valid.append(False)
            continue
        computed = _evaluate_binary(left, operator, right)
        if computed != result:
            valid.append(False)
            continue
        available.subtract(used)
        available += Counter()
        available[result] += 1
        last_result = result
        valid.append(True)
    correct = (
        len(equations) == num_equations
        and all(valid)
        and last_result == int(target)
    )
    return CountdownCheck(
        correct=correct,
        valid_equations=tuple(valid),
        prediction=prediction,
    )


def check_countdown_solution_nextlat_compat(
    *,
    input_numbers: Sequence[int],
    target: int,
    prediction: str,
    num_equations: int,
) -> CountdownCheck:
    """Match the looser upstream NextLat Countdown evaluator semantics."""
    equations = prediction.split(",") if prediction else []
    available = {int(number) for number in input_numbers}
    valid: list[bool] = []
    last_result: int | None = None
    for index in range(num_equations):
        if index >= len(equations):
            valid.append(False)
            continue
        match = _EQUATION_RE.match(equations[index])
        if match is None:
            valid.append(False)
            continue
        left = int(match.group(1))
        operator = match.group(2)
        right = int(match.group(3))
        result = int(match.group(4))
        if left not in available or right not in available:
            valid.append(False)
            continue
        computed = _evaluate_binary(left, operator, right)
        if computed != result:
            valid.append(False)
            continue
        available.add(result)
        last_result = result
        valid.append(True)
    correct = (
        len(equations) == num_equations
        and all(valid)
        and last_result == int(target)
    )
    return CountdownCheck(
        correct=correct,
        valid_equations=tuple(valid),
        prediction=prediction,
    )
