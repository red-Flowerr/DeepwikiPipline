import argparse
import json
from pathlib import Path


def load_narratives(source: Path) -> list[str]:
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    narratives: list[str] = []
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                narrative = entry.get("narrative")
                if isinstance(narrative, str):
                    narratives.append(narrative.strip())
    return narratives


def write_output(narratives: list[str], destination: Path) -> None:
    destination.write_text("\n\n".join(filter(None, narratives)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export concatenated narratives from DeepWiki results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("result_data/verl_narratives.json"),
        help="Path to the JSON file produced by the pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result_data/verl_narratives_v2.txt"),
        help="Path to write the concatenated narratives.",
    )
    args = parser.parse_args()
    narratives = load_narratives(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_output(narratives, args.output)
    print(f"Wrote {len(narratives)} narratives to {args.output}")


if __name__ == "__main__":
    main()
