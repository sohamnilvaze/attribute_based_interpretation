import json
from collections import defaultdict
import numpy as np

from llm import query_llm
from attribute_parsers import parse_attributes


class RepeatedSingleParticle:
    def __init__(
        self,
        entities: dict[str, str],
        attributes: list[str],
        prompts: dict[str, str],
        n_runs: int
    ):
        self.entities = entities
        self.attributes = attributes
        self.prompts = prompts
        self.n_runs = n_runs

    def run(self):
        results = defaultdict(list)

        for entity_type, entity_name in self.entities.items():
            for attr in self.attributes:
                print(f"For attribute: {attr}")
                prompt_template = self.prompts[attr]

                for i in range(self.n_runs):
                    print(f"Run: {i + 1}")

                    prompt = prompt_template.format(entity=entity_name)

                    output = query_llm(
                        prompt,
                        tag=f"baseline_{entity_type}_{attr}"
                    )

                    parsed = parse_attributes(output)

                    results[f"{entity_type}_{attr}"].append({
                        "run_id": i,
                        "prompt": prompt,
                        "raw_output": output,
                        "parsed_value": parsed.get(attr, None)
                    })

        with open("../data/baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)

    def analyze(self):
        def summarize(values):
            clean = [v for v in values if v is not None]

            if len(clean) == 0:
                return {
                    "mean": None,
                    "std": None,
                    "missing_rate": 1.0
                }

            return {
                "mean": float(np.mean(clean)),
                "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
                "missing_rate": 1 - len(clean) / len(values)
            }

        with open("../data/baseline_results.json") as f:
            results = json.load(f)

        summary = {}

        for key, runs in results.items():
            values = [r["parsed_value"] for r in runs]
            summary[key] = summarize(values)

        with open("../data/baseline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))

entities = {
    "real": "Brad Pitt",
    "fake": "Cameron Ridgewell"
}

attrs = ["skin_color", "profession_actor", "age"]

PROMPTS = {
    "skin_color": "What is the skin color of {entity}? Answer in one short sentence.",
    "profession_actor": "Is {entity} an actor? Answer in one short sentence.",
    "age": "What is the age of {entity}? Answer in one short sentence."
}

exp = RepeatedSingleParticle(
    entities = entities,
    attributes = attrs,
    prompts = PROMPTS,
    n_runs = 1
)

exp.run()
exp.analyze()
