'''
Skin color: 0.0 = very dark, 0.5 = medium, 1.0 = very light
Profession: Actor likelihood: 0.0 = definitely not an actor, 0.5 = meybe/ unclear, 1.0 = definitely an actor
Age (normalized): age_real E [0,100] age_norm = age_real/100
'''

import re

# Skin color parser
def parse_skin_color(text: str):
    text = text.lower()

    light_terms = ["white", "fair", "light-skinned", "light skin"]
    medium_terms = ["medium", "olive", "tan"]
    dark_terms = ["dark", "black", "dark-skinned"]

    for t in light_terms:
        if t in text:
            return 0.9

    for t in medium_terms:
        if t in text:
            return 0.5

    for t in dark_terms:
        if t in text:
            return 0.1

    return None

#Professional likelihood parser
def parse_actor_likelihood(text: str):
    text = text.lower()

    if "actor" in text or "film star" in text or "movie star" in text:
        return 1.0

    if "entertainment" in text or "celebrity" in text:
        return 0.6

    if "not an actor" in text:
        return 0.0

    return None

# Age parser
def parse_age(text: str):
    text = text.lower()

    # Explicit number
    match = re.search(r"(\d{2})\s*(years|year)?", text)
    if match:
        age = int(match.group(1))
        return min(age / 100.0, 1.0)

    # Heuristic buckets
    if "young" in text:
        return 0.25
    if "middle-aged" in text:
        return 0.5
    if "older" in text or "old" in text:
        return 0.7

    return None

# unified parsing interface
def parse_attributes(text: str):
    return {
        "skin_color": parse_skin_color(text),
        "profession_actor": parse_actor_likelihood(text),
        "age": parse_age(text)
    }

# json parser for method2
import json

def safe_parse_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None


# samples = [
#     "Brad Pitt is a white American actor in his early 60s.",
#     "He is a famous Hollywood movie star.",
#     "I don't know who this person is."
# ]

# for s in samples:
#     print(s)
#     print(parse_attributes(s))
#     print("----")
