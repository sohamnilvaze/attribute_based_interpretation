# Attribute Based Interpretation (ABI)
Repository for the ABI method, used for **attribute-level interpretability** in foundation models.

This method identifies how a model assigns interpretable attributes (e.g., skin color, age, profession likelihood) to a entity (e.g. Brad Pitt), enabling black-box gradient-free explainability and behavior analysis during model adaptation (including machine unlearning, transfer learning, knowledge editing and continous learning).

## Context

Explainability is a central requirement for the responsible development and deployment of foundation models because their scale, generality, and opacity amplify both their impact and their risks. Beyond risk mitigation, explainability supports scientific progress by allowing researchers to form hypotheses about learned representations and training dynamics, thereby turning foundation models from purely empirical artifacts into objects of systematic study.

Most explainability methods perform input attribution, that is, assigning importance scores to input features that drove a model's output [[Simonyan2014](https://arxiv.org/abs/1409.1556)]. For example, in language models techniques like Integrated Gradients can show which tokens most influenced a next-token prediction, while in image-generation models attention-rollout can highlight which regions of a generated image are most related to specific prompt words.

Despite their popularity, pure input-attribution methods such as saliency maps and simple heatmaps have significant limitations, for instance highlighting regions correlated with a prediction without revealing the internal causal mechanisms that produced the output [[Paez2025](https://link.springer.com/chapter/10.1007/978-3-032-03083-2_7)]. Moreover, it is often difficult for domain specialists to assess a model’s knowledge based solely on input attribution, because experts often reason about meaningful higher-level attributes [[Malach2025](https://link.springer.com/article/10.1007/s10994-025-06852-8)].

Several method attempt to overcome those challenges by providing explanations that go beyong input attribution, such as: TCAV uses directional derivatives to quantify the degree to which a user-defined concept is important to a classification result [[Kim2018](https://arxiv.org/abs/1711.11279)]; KnowProb probe whether black-box models understand implicit knowledge beyond the given text [[Zhao2025](https://arxiv.org/html/2508.16969v1)]; LAMA is a probe for analyzing the factual and commonsense knowledge contained in pretrained language models [[Petroni2019](https://arxiv.org/abs/1909.01066)]; Last but not least, the work of Tinaz et al. explore the evolution of visual representations throughout the diffusion process [[Tinaz2026](https://arxiv.org/pdf/2504.15473)].

## Goal

This work develops a novel interpretability method, addressing the following gap in the state-of-the-art: explaining how the generated content leverages the model's internal knowledge in terms of high-level and human-friendly attributes. For example, when prompting a text-to-image generative model for "a picture of Brad Pitt", providing an explanation of the sort "the generated image is such because the model knows he is a white american actor". To the best of our knowledge, no existing method is able to provide such explanations in a black-box, architecture-agnostic, gradient-free, and adaptation-friendly manner.

Additionally, the outputs of ABI can be easily used for other downstream tasks beyond interpretability by the end user:
- Quantify the model's attribute-level variance
- Establish a baseline behavioral profile
- Provide supervision signal for bias-correcting finetunning
- Probe unlearned models for residual knowledge and/or adversarial attack evaluation


# How ABI works

1. Given a prompt, identify named entities (for example, Brad Pitt), using either: an auxiliary language model (that is NOT the model being tested, but instead a separate well known model such as GPT4), a hardcoded table (mainly only for experimentation purposes), or a human-in-the-loop (for deployed systems).

2. Identify attributes of interest for that entity, using a similar method as above.

3. Inquire what a model still knows about the entity, in terms of which values it assign to the identified attributes (thus a human understandable explanation of the residual knowledge). Even if the model refuses to generate content related the entire prompt, we may still be “leak” information about some of the entitiy's attributes (for example, Brad Pitt's skin color, profession, etc). We call this the "knowledge probing" step, and we intend to experiment with several methods. In contrast with basic knowledge extraction methods such as LAMA [[Petroni2019](https://arxiv.org/abs/1909.01066)], we assume this knowledge can not be trivially extracted from the model (for example, it may refuse due to a previous unlearning training, or its knowledge may be partial due to continual learning), and a indirect/iterative/repetitive "interrogation" method may help to do so. Regardless of how a method work, at the end of this step it must assign a single value of it attribute. Currently, the tested methods are:
    * **Simple baseline**: Just ask the model directly 🙂
    * **Still-quite-simple baseline**: Ask the model directly, but repeat several times.
    * **Literature baseline**: find if there is some method that could be used for this (maybe LAMA?) and see how they perform.
    * **Particle-based**: see in specific section. Unless otherwise noted, this is the knowledge probing method used when we mention ABI.

4. Return the attribute-based explanation to the user

## Particle-Based Knowledge Probing

This knowledge probing method uses the idea of "crowd intelligence", in which simple independent agents/particles produces behaviors that are superior to those of individual agents, aiming to extract from the model what it knowns about an entity, even when the model seems to refuse a direct answer.

In order to perform that, it instantiates several "particles" (each one defined by a simple state, its current estimate of the attributes) that are "simulated" over time (repitedly trying to refine their own estimates) aiming to achieve some "consensus" (the average or majority vote of the entire set of particles).

Pseudo code:

```
Inputs = entity_name: str, number_of_iterations: int, number_of_particles: int, attribute_names: List[str]
internal variables = state_of_each_particle: List[Dict[str, float]], type_of_each_particle: List[Type[ParticleType]]

for number_of_iterations iterations:
    for number_of_particles particles: (their first estimate initialized randomly, uniformly over the scale space)
        ask each agent to estimate the attribtes, passing based on its last estimate and the centroid; mapping output-to-attribute
    calculate centroid of estimates (or maybe an independent centroid for each agent, with the nearest)
analize convergence, determine final estimate
Outputs = attribute_values: List[float]
```

The mapping output-to-attribute is the only architecture-dependent component of ABI. It abstracts the exact prompting and output interpretation that is needed for each architecture. The mapping method is defined by the class ParticleType, that performs all communication with the actual underlying models. Currently, we have the following mapping methods:
* For language models, just ask and postprocess the text output.
* For text to image model... we are not sure yet... See the TODO list...

# Experimentation methodology


We focus on **interpretable scalar attributes** extracted from the outputs of natural language small foundation model

From now on I will refer to “task” as what we are trying to understand from the LLM.
It does NOT refer to *what you have to do as researcher*, like what is being described here.
A task is composed of a set of entities, each of them annotated with several ground truth attributes.
For example, consider the hypothetical task of "investigating people", in which each entity is annotated with 3 attributes (skin color, profession, age), and one such entity is Brad Pitt.

## Attributes Modeled

Each attribute is normalized to `[0,1]`:

| Attribute | Description | Scale |
|------------|------------|--------|
| Skin Color | Estimated skin tone | 0.0 = very dark, 0.5 = medium, 1.0 = very light |
| Profession (Actor) | Likelihood of being an actor | 0.0 = definitely not, 0.5 = unclear, 1.0 = definitely |
| Age | Normalized age | `age_real / 100` |

# TODOs
This is a work in progress
We welcome contrbutions and pull requests :)

Specific work packages:

- [x] Choose the LLM, run it locally

- [x] Define 2 very simple tasks: (1) the brad pitt task described above (2) a similar task about Cameron Ridgewell (this person doesnt exist, so it gives us a glimpse of what will the LLM do once we unlearn Brad Pitt). Step 1 and 2 of ABI can be fully hardcoded.

- [x] Implement the two baseline knowledge probing methods

- [ ] Implement Particle-Based Knowledge Probing, with the "mapping output-to-attribute" as a clearly separated function/class, and currently implemented only for language models (which is trivially simple, and requires at most proper prompting and some output filtering)

- [ ] See if we get any interesting results
  * Just observe the trajectory of the many particles to see if there is any distinguishable pattern (converges, oscilates, etc)
  * See if the final centroid concides with the actual attribute of that entity
  * When prompted with the additional information of these attributes, does the model "recover the memory" about the entity?

- [ ] Design and implement the "mapping output-to-attribute" for text-to-image models
  * Maybe substitute the direct estimate of attributes by predicting an image and then seeing its similarity with the target usign CLIP? This this case maybe would make more sense to just go slowly building a estimation of the "atributes -> CLIP" function?
  * Or maybe identifying (some svd dimentionallity reduction style) which directions in the vector space correlate most with the attributes?

- [ ] Repeat evaluation using Vision-Unlearning basic testbed (no need to define new tasks or finetune models, this is already provided by the lib)

- [ ] Analyze the results more carefully, publish paper

- [ ] Automate step 1 and 2 of the method



If you use the code in this repository, the ideas invoved in ABI, or any other significant aspect of this project, please cite us:
```
@misc{attribute_based_interpretation,
  author       = {Soham Vase, Leonardo Santiago Benitez Pereira},
  title        = {attribute_based_interpretation},
  howpublished = {\url{[https://github.com/LeonardoSanBenitez/sparse-peft](https://github.com/sohamnilvaze/attribute_based_interpretation)}},
  note         = {Accessed: 2026-02-11},
  year         = {2026}
}
```
