from llm import query_llm
from attribute_parsers import safe_parse_json


class ParticleType:
    """
    Architecture-dependent interface.
    Any model-specific probing must inherit from this.
    """

    def query(self, entity, particle_state, centroid, memory=None):
        raise NotImplementedError

    def map_output_to_attributes(self, raw_output, previous_state):
        raise NotImplementedError


class LanguageModelParticle(ParticleType):
    """
    Implementation for language models via prompting.
    """

    def build_prompt(self, entity, particle, centroid, memory=None):
        prompt = f"""
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

        if memory is not None:
            prompt += f"""

In the previous step, you answered:
- Skin color: {memory['skin_color']:.2f}
- Actor likelihood: {memory['profession_actor']:.2f}
- Age: {memory['age']:.2f}

Decide whether to keep or revise your previous estimate.
"""

        prompt += f"""

Using your knowledge, refine the attributes for {entity}.

Output ONLY valid JSON with keys:
skin_color, profession_actor, age.
"""

        return prompt.strip()

    def query(self, entity, particle_state, centroid, memory=None):
        prompt = self.build_prompt(entity, particle_state, centroid, memory)
        return query_llm(prompt, tag="particle_probe")

    def map_output_to_attributes(self, raw_output, previous_state):
        parsed = safe_parse_json(raw_output)

        if not parsed:
            return previous_state

        updated = {}
        for k in previous_state.keys():
            try:
                v = float(parsed[k])
                updated[k] = min(max(v, 0.0), 1.0)
            except:
                updated[k] = previous_state[k]

        return updated
