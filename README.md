# Attribute Based Interpretation

A lightweight experimental framework for measuring **attribute-level behavioral instability** in small language models.

This project evaluates how consistently a model assigns interpretable attributes (e.g., skin color, age, profession likelihood) across repeated runs, and provides tools to analyze baseline behavior before applying unlearning or adversarial methods.

---

## Project Goal

The goal of this project is to:

- Quantify **attribute-level interpretation variance**
- Establish a **baseline behavioral profile**
- Provide infrastructure for future:
  - Adversarial unlearning
  - Behavior modification
  - Representation manipulation
  - Stability comparison studies

We focus on **interpretable scalar attributes** extracted from natural language model outputs.

---

## Attributes Modeled

Each attribute is normalized to `[0,1]`:

| Attribute | Description | Scale |
|------------|------------|--------|
| Skin Color | Estimated skin tone | 0.0 = very dark, 0.5 = medium, 1.0 = very light |
| Profession (Actor) | Likelihood of being an actor | 0.0 = definitely not, 0.5 = unclear, 1.0 = definitely |
| Age | Normalized age | `age_real / 100` |

---

## 🏗 Project Structure

