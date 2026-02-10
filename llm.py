import ollama
import json
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path("../config/model_config.json")
LOG_DIR = Path("../data/raw_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

MODEL = CONFIG["model"]

def query_llm(prompt: str, tag: str = "default") -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": CONFIG["temperature"],
            "top_p": CONFIG["top_p"],
            "top_k": CONFIG["top_k"],
            "repeat_penalty": CONFIG["repeat_penalty"],
            "num_predict": CONFIG["max_tokens"],
        }
    )

    output = response["message"]["content"]
    timestamp = datetime.utcnow().isoformat()

    # log = {
    #     "timestamp": timestamp,
    #     "model": MODEL,
    #     "prompt": prompt,
    #     "output": output,
    #     "config": CONFIG,
    #     "tag": tag
    # }

    # log_file = LOG_DIR / f"{timestamp.replace(':','_')}.json"
    # with open(log_file, "w") as f:
    #     json.dump(log, f, indent=2)

    return output
