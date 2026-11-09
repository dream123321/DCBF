from __future__ import annotations

import argparse


COMMAND_EXAMPLES = {
    "train": "dcbf train train.xyz",
    "efs-distri": "dcbf efs-distri data.xyz",
    "predict-xyz": (
        "dcbf predict-xyz input.xyz --calc-type sus2 "
        "--model current.sus2 --elements Si"
    ),
    "plot-errors": "dcbf plot-errors dft.xyz mlip.xyz",
    "coverage-pca": "dcbf coverage-pca --input sampled_training_dataset.xyz --query query.xyz --model current.sus2",
    "create-init": "dcbf create-init",
    "mp-search": "dcbf mp-search Li P S Cl --api-key YOUR_API_KEY",
    "run": "dcbf run dcbf.json",
    "reduce": "dcbf reduce reduce.json",
    "raw-dft": "dcbf raw-dft pack task_1",
    "kill": "dcbf kill workspace",
}

ADVANCED_COMMAND_EXAMPLES = {
    "run-generation": "dcbf run-generation --workspace workspace/main_0/gen_0",
    "benchmark-selection": "dcbf benchmark-selection",
    "calibrate-selection": "dcbf calibrate-selection",
}


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def command_example_epilog(command: str) -> str:
    return f"Example:\n  {COMMAND_EXAMPLES[command]}"


def advanced_command_example_epilog(command: str) -> str:
    return f"Example:\n  {ADVANCED_COMMAND_EXAMPLES[command]}"


def append_command_example(text: str, command: str) -> str:
    return f"{text}\n\n{command_example_epilog(command)}" if text else command_example_epilog(command)


def top_level_epilog(include_advanced_commands: bool) -> str:
    sections = []
    if not include_advanced_commands:
        sections.append("Use `dcbf -hh` to show advanced commands.")

    sections.append(
        "Examples:\n"
        + "\n".join(f"  {example}" for example in COMMAND_EXAMPLES.values())
    )
    if include_advanced_commands:
        sections.append(
            "Advanced examples:\n"
            + "\n".join(
                f"  {example}" for example in ADVANCED_COMMAND_EXAMPLES.values()
            )
        )
    return "\n\n".join(sections)
