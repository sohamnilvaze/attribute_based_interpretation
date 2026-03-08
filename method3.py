import copy
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from particle_types import LanguageModelParticle


model = "qwen1.8b"


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
# Experiment runner (Method 3)
# =========================

class ReflectiveCrowdExperiment:

    def __init__(
        self,
        n_particles: int,
        n_runs: int,
        attributes: List[str],
        entity_names: Dict[str, str],
        particle_type
    ):
        self.n_particles = n_particles
        self.n_runs = n_runs
        self.attributes = attributes
        self.entity_names = entity_names
        self.analysis_res = {}
        self.particle_type = particle_type

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

                previous_answer = p if step > 0 else None

                raw_output = self.particle_type.query(
                    entity_name,
                    p,
                    centroid,
                    memory=previous_answer
                )

                updated_particle = self.particle_type.map_output_to_attributes(
                    raw_output,
                    p
                )

                new_particles.append(updated_particle)

            particle.update_particles(new_particles)

        return trajectory

    def run_method(self):
        for label, entity in self.entity_names.items():
            traj = self.run(entity, label)
            with open(f"../data/trajectory3_{label}.json", "w") as f:
                json.dump(traj, f, indent=2)

    # =========================
    # Analysis
    # =========================

    def summarize_all(self, epsilon=0.01, patience=3):

        def summarize(traj):
            centroids = [step["centroid"] for step in traj]
            result = {}

            for attr in centroids[0].keys():
                values = [c[attr] for c in centroids]
                diffs = np.diff(values)
                stepwise_change = np.abs(diffs)

                converged_step = None
                stable_count = 0
                for i, change in enumerate(stepwise_change):
                    if change < epsilon:
                        stable_count += 1
                        if stable_count >= patience:
                            converged_step = i
                            break
                    else:
                        stable_count = 0

                result[attr] = {
                    "initial_value": values[0],
                    "final_value": values[-1],
                    "delta_total": values[-1] - values[0],
                    "std_over_time": float(np.std(values)),
                    "mean_stepwise_change": float(np.mean(stepwise_change)) if len(stepwise_change) > 0 else 0.0,
                    "converged_step": converged_step,
                    "stability_score": float(1 / (1 + np.mean(stepwise_change))) if len(stepwise_change) > 0 else 1.0
                }
            with open(f"../data/summary_{model}_{label}.json", "w") as f:
                json.dump(result, f, indent=2)
            return result

        for label in self.entity_names.keys():
            print(f"\nSummarizing for {label}")
            with open(f"../data/trajectory3_{label}.json") as f:
                traj = json.load(f)
                print(summarize(traj))

    # =========================
    # Plots
    # =========================

    def plot_centroid(self):
        for label in self.entity_names.keys():
            centroids, _, _ = load_centroids(f"../data/trajectory3_{label}.json")

            plt.figure()
            for attr in self.attributes:
                plt.plot(centroids[attr], label=attr)

            plt.title(f"Centroid trajectory (Method3 - {label})")
            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"../{model}_{label}_centroid_trajectory3.png")
            plt.show()

    def plot_variance(self):
        for label in self.entity_names.keys():
            _, variances, _ = load_centroids(f"../data/trajectory3_{label}.json")

            plt.figure()
            for attr in self.attributes:
                plt.plot(variances[attr], label=attr)

            plt.title(f"Particle variance (Method3 - {label})")
            plt.xlabel("Iteration")
            plt.ylabel("Variance")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"../{model}_{label}_variance3.png")
            plt.show()
    
    def plot_particle_scatter_grid(self, x_attr="skin_color", y_attr="age"):

        for label in self.entity_names.keys():

            with open(f"../data/trajectory3_{label}.json") as f:
                traj = json.load(f)

            n_steps = len(traj)

            cols = 5
            rows = int(np.ceil(n_steps / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            axes = axes.flatten()

            for i, step in enumerate(traj):

                particles = step["particles"]

                x = [p[x_attr] for p in particles]
                y = [p[y_attr] for p in particles]

                axes[i].scatter(x, y)

                axes[i].set_title(f"Step {step['step']}")
                axes[i].set_xlabel(x_attr)
                axes[i].set_ylabel(y_attr)

                axes[i].set_xlim(0,1)
                axes[i].set_ylim(0,1)

            # hide unused axes
            for j in range(i+1, len(axes)):
                axes[j].axis("off")

            fig.suptitle(f"Particle Distribution Over Time ({label})", fontsize=16)

            plt.tight_layout()

            plt.savefig(f"plots/{model}_{label}_particle_scatter_grid.png")

            plt.show()

entities = {
    "real": "Brad Pitt",
    "fake": "Cameron Ridgewell"
}

attrs = ["skin_color", "profession_actor", "age"]

particle_type = LanguageModelParticle()  # optional flag

exp3 = ReflectiveCrowdExperiment(
    n_particles=10,
    n_runs=10,
    attributes=attrs,
    entity_names=entities,
    particle_type=particle_type
)

# exp3.run_method()
# exp3.plot_centroid()
# exp3.plot_variance()
exp3.plot_particle_scatter_grid()
# exp3.summarize_all()