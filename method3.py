from llm import query_llm
from attribute_parsers import safe_parse_json
import copy
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


# =========================
# Prompt
# =========================

def particle_update_prompt(entity, particle, centroid, previous_answer=None):
    base_prompt = f"""
You are refining estimates of attributes for a person.

Current estimate:
- Skin color (0=dark, 1=light): {particle['skin_color']:.2f}
- Actor likelihood (0=no, 1=yes): {particle['profession_actor']:.2f}
- Age (0=young, 1=old): {particle['age']:.2f}

Group average estimate:
- Skin color: {centroid['skin_color']:.2f}
- Actor likelihood: {centroid['profession_actor']:.2f}
- Age: {centroid['age']:.2f}
"""

    if previous_answer is not None:
        base_prompt += f"""

In the previous step, you answered:
- Skin color: {previous_answer['skin_color']:.2f}
- Actor likelihood: {previous_answer['profession_actor']:.2f}
- Age: {previous_answer['age']:.2f}

Decide whether to:
1. Keep the previous answer (if you believe it is stable), OR
2. Modify it based on better reasoning.

If modifying, explain briefly to yourself internally but output ONLY the updated JSON.
"""

    base_prompt += f"""

Using your knowledge and reasoning, provide the updated values for {entity}.

Output ONLY valid JSON with keys:
skin_color, profession_actor, age.
"""

    return base_prompt.strip()

# =========================
# Utilities
# =========================

def load_centroids(path):
    with open(path) as f:
        traj = json.load(f)

    centroids = {k: [] for k in traj[0]["centroid"].keys()}
    variances = {k: [] for k in traj[0]["centroid"].keys()}

    for step in traj:
        particles = step["particles"]
        for attr in centroids.keys():
            values = [p[attr] for p in particles]
            centroids[attr].append(np.mean(values))
            variances[attr].append(np.var(values))

    return centroids, variances, traj


# =========================
# Particle container
# =========================

class Particle:
    def __init__(self, attributes: List[str], n_particles: int):
        self.attributes = attributes
        self.n_particles = n_particles
        self.particles = []
        self.centroid = {}

    def random_particle(self):
        return {attr: random.random() for attr in self.attributes}

    def initialize_particles(self):
        self.particles = [self.random_particle() for _ in range(self.n_particles)]
        return self.particles

    def update_particles(self, new_particles):
        self.particles = new_particles

    def compute_centroid(self):
        centroid = {}
        for attr in self.attributes:
            centroid[attr] = sum(p[attr] for p in self.particles) / len(self.particles)
        self.centroid = centroid
        return centroid


# =========================
# Experiment runner
# =========================

class CrowdIntelligence:
    def __init__(
        self,
        n_particles: int,
        n_runs: int,
        attributes: List[str],
        entity_names: Dict[str, str]
    ):
        self.n_particles = n_particles
        self.n_runs = n_runs
        self.attributes = attributes
        self.entity_names = entity_names
        self.analysis_res = {}

    def run(self, entity_name: str, label: str):
        particle = Particle(self.attributes, self.n_particles)
        particle.initialize_particles()

        trajectory = []

        print(f"\nRunning entity [{label}]: {entity_name}")

        for step in range(self.n_runs):
            print(f"  Step {step}")

            centroid = particle.compute_centroid()

            trajectory.append({
                "step": step,
                "particles": copy.deepcopy(particle.particles),
                "centroid": centroid
            })

            new_particles = []

            for p in particle.particles:
                previous_answer = None
                if step > 0:
                    previous_answer = p
                prompt = particle_update_prompt(
                    entity_name,
                    p,
                    centroid,
                    previous_answer=previous_answer
                )
                output = query_llm(
                    prompt,
                    tag=f"method2_{label}_step{step}"
                )

                parsed = safe_parse_json(output)

                if parsed:
                    updated = {}
                    for k in p.keys():
                        try:
                            v = float(parsed[k])
                            updated[k] = min(max(v, 0.0), 1.0)
                        except:
                            updated[k] = p[k]
                    new_particles.append(updated)
                else:
                    new_particles.append(p)

            particle.update_particles(new_particles)

        return trajectory

    def run_method(self):
        for label, entity in self.entity_names.items():
            traj = self.run(entity, label)
            with open(f"../data/trajectory_{label}.json", "w") as f:
                json.dump(traj, f, indent=2)

    # =========================
    # Analysis
    # =========================

    def analyze(self):
        def summarize(traj):
            centroids = [step["centroid"] for step in traj]
            result = {}

            for attr in centroids[0].keys():
                values = [c[attr] for c in centroids]
                result[attr] = {
                    "final_value": values[-1],
                    "std_over_time": float(np.std(values)),
                    "delta": values[-1] - values[0],
                    "stepwise_change_mean": float(np.mean(np.abs(np.diff(values))))
                }

            return result

        for label in self.entity_names.keys():
            with open(f"../data/trajectory_{label}.json") as f:
                traj = json.load(f)
            self.analysis_res[label] = summarize(traj)

        print(json.dumps(self.analysis_res, indent=2))

    # =========================
    # Plots
    # =========================

    def plot_variance(self):
        for label in self.entity_names.keys():
            _, variances, _ = load_centroids(f"../data/trajectory_{label}.json")
            plt.figure()
            for attr in self.attributes:
                plt.plot(variances[attr], label=attr)
            plt.title(f"Particle variance over time ({label})")
            plt.xlabel("Iteration")
            plt.ylabel("Variance")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_centroid(self):
        for label in self.entity_names.keys():
            centroids, _, _ = load_centroids(f"../data/trajectory_{label}.json")
            plt.figure()
            for attr in self.attributes:
                plt.plot(centroids[attr], label=attr)
            plt.title(f"Centroid trajectory ({label})")
            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_final_particles(self):
        for label in self.entity_names.keys():
            with open(f"../data/trajectory_{label}.json") as f:
                traj = json.load(f)

            final_particles = traj[-1]["particles"]

            plt.figure(figsize=(10, 3))
            for i, attr in enumerate(self.attributes):
                values = [p[attr] for p in final_particles]
                plt.subplot(1, len(self.attributes), i + 1)
                plt.hist(values, bins=10)
                plt.title(attr)
                plt.ylim(0, len(values))

            plt.suptitle(f"Final particle distributions ({label})")
            plt.tight_layout()
            plt.show()


entities = {
    "real": "Brad Pitt",
    "fake": "Cameron Ridgewell"
}

attrs = ["skin_color", "profession_actor", "age"]

exp = CrowdIntelligence(
    n_particles=10,
    n_runs=5,
    attributes=attrs,
    entity_names=entities
)

exp.run_method()
exp.analyze()
