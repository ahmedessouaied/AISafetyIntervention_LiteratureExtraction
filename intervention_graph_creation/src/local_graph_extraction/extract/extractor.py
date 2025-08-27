import os
import re
import json
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from config import load_settings
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT
from intervention_graph_creation.src.local_graph_extraction.extract.utilities import (safe_write,
                                                                                      split_text_and_json,
                                                                                      stringify_response,
                                                                                      extract_output_text,
                                                                                      write_failure,
                                                                                      url_to_id,
                                                                                      filter_dict)

# ---------------- Basic config ----------------


MODEL = "o3"
REASONING_EFFORT = "medium"
SETTINGS = load_settings()
META_KEYS = frozenset(['authors', 'date_published', 'filename', 'source', 'source_filetype', 'title', 'url'])


class Extractor:
    """Upload PDF -> call model -> save raw/summary/json."""
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

    def upload_pdf_get_id(self, pdf_path: Path) -> str:
        """Upload a PDF and return its file id."""
        try:
            with pdf_path.open("rb") as fh:
                f = self.client.files.create(file=fh, purpose="user_data")
            return f.id
        except FileNotFoundError:
            raise
        except PermissionError:
            raise
        except OSError as e:
            raise OSError(f"I/O error while reading '{pdf_path}': {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to upload to OpenAI: {e}") from e

    def call_openai_file(self, file_id: str) -> Any:
        """Call the model with the file id and return the raw response."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": PROMPT_EXTRACT},
            ],
        }]
        try:
            return self.client.responses.parse(
                model=MODEL,
                input=messages,
                reasoning={"effort": REASONING_EFFORT},
            )
        except Exception as e:
            # One simple standard type for API-level failures
            raise RuntimeError(f"OpenAI API call failed: {e}") from e
        
    def call_openai_text(self, file_text: str) -> Any:
        """Call the model with the file text and return the raw response."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT_EXTRACT},
                {"type": "input_text", "text": f"\n\nHere is the paper for analysis:\n\n{file_text}"},
            ],
        }]
        try:
            return self.client.responses.parse(
                model=MODEL,
                input=messages,
                reasoning={"effort": REASONING_EFFORT},
            )
        except Exception as e:
            # One simple standard type for API-level failures
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def write_outputs(self, out_dir: Path, stem: str, resp: Any, meta: json) -> None:
        """Write raw response, summary text, and parsed JSON (or raise)."""
        raw_path = out_dir / f"{stem}_raw_response.txt"
        json_path = out_dir / f"{stem}.json"
        summary_path = out_dir / f"{stem}_summary.txt"

        safe_write(raw_path, stringify_response(resp))

        output_text = extract_output_text(resp)
        text_part, json_part = split_text_and_json(output_text)

        safe_write(summary_path, text_part or "")

        if json_part is None:
            raise ValueError("No JSON block found in output_text")

        parsed = json.loads(json_part)

        if meta:
            parsed['meta'] = meta

        safe_write(json_path, json.dumps(parsed, ensure_ascii=False, indent=2))

    def process_pdf(self, path: Path) -> None:
        """Process a single PDF."""
        out_dir = SETTINGS.paths.output_dir / path.stem
        if out_dir.exists():
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        file_id = self.upload_pdf_get_id(path)
        resp = self.call_openai_file(file_id)
        meta = [{"key": "filename", "value": path.name}] # other meta?

        self.write_outputs(out_dir, path.stem, resp, meta)

    def process_jsonl(self, path: Path) -> None:
        """Process a single JSONL file."""
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    paper_json = json.loads(line)

                    paper_id = path.stem + "__" + url_to_id(paper_json.get('url', 'unknown'))
                
                    out_dir = SETTINGS.paths.output_dir / paper_id
                    if out_dir.exists():
                        return
                    out_dir.mkdir(parents=True, exist_ok=True)

                    resp = self.call_openai_text(paper_json['text'])
                    meta = filter_dict(paper_json, META_KEYS)

                    self.write_outputs(out_dir, paper_id, resp, meta)

        except Exception as e:
            raise RuntimeError(f"Failed to read JSON file '{path}': {e}") from e

    def process_dir(self, input_dir: Path, first_n: Optional[int] = None) -> None:
        """Process all files in a directory. Single catch-all per file for diagnostics."""
        SETTINGS.paths.output_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for ext in ('*.pdf', '*.jsonl'):
            files.extend(input_dir.glob(ext))

        if first_n:
            files = files[:first_n]

        print(f"Found {len(files)} files in {input_dir} to process.")
        print(files)

        for file in tqdm(files):
            if file.suffix.lower() == '.jsonl':
                try:
                    self.process_jsonl(file)
                except Exception as e:
                    write_failure(SETTINGS.paths.output_dir, file.stem, e)
                continue
            elif file.suffix.lower() == '.pdf':
                try:
                    self.process_pdf(file)
                except Exception as e:
                    write_failure(SETTINGS.paths.output_dir, file.stem, e)
                continue


if __name__ == "__main__":
    extractor = Extractor()
    extractor.process_dir(SETTINGS.paths.input_dir, 2)
