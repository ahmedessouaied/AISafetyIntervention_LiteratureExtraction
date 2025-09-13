import os
import json
import time
from pathlib import Path
from typing import Any, Optional, List, Dict

from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from config import load_settings
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT
from intervention_graph_creation.src.local_graph_extraction.extract.utilities import (
    safe_write,
    split_text_and_json,
    stringify_response,
    extract_output_text,
    write_failure,
    url_to_id,
    filter_dict,
)

MODEL = "o3"
REASONING_EFFORT = "medium"
SETTINGS = load_settings()
META_KEYS = frozenset(
    [
        "authors",
        "date_published",
        "filename",
        "source",
        "source_filetype",
        "title",
        "url",
    ]
)


class Extractor:
    """Upload PDF -> call model -> save raw/summary/json."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

        self._n = 0
        self._sum_sec = 0.0
        self._sum_in = 0
        self._sum_out = 0
        self._sum_tot = 0

    def upload_pdf_get_id(self, pdf_path: Path) -> str:
        with pdf_path.open("rb") as fh:
            f = self.client.files.create(file=fh, purpose="user_data")
        return f.id

    def call_openai_file(self, file_id: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": PROMPT_EXTRACT},
                ],
            }
        ]
        return self.client.responses.parse(
            model=MODEL,
            input=messages,
            reasoning={"effort": REASONING_EFFORT},
        )

    def call_openai_text(self, file_text: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT_EXTRACT},
                    {
                        "type": "input_text",
                        "text": f"\n\nHere is the paper for analysis:\n\n{file_text}",
                    },
                ],
            }
        ]
        return self.client.responses.parse(
            model=MODEL,
            input=messages,
            reasoning={"effort": REASONING_EFFORT},
        )

    def write_outputs(self, out_dir: Path, stem: str, resp: Any, meta: json) -> None:
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
            parsed["meta"] = meta

        safe_write(json_path, json.dumps(parsed, ensure_ascii=False, indent=2))

    def _usage_from_resp(self, resp: Any):
        try:
            usage = getattr(resp, "usage", None)
            if isinstance(usage, dict):
                tin = int(usage.get("input_tokens", 0))
                tout = int(usage.get("output_tokens", 0))
                ttot = int(usage.get("total_tokens", tin + tout))
                return tin, tout, ttot
        except Exception:
            pass
        try:
            as_dict = resp.dict() if hasattr(resp, "dict") else {}
            if as_dict:
                u = as_dict.get("usage", {}) or {}
                if u:
                    tin = int(u.get("input_tokens", 0))
                    tout = int(u.get("output_tokens", 0))
                    ttot = int(u.get("total_tokens", tin + tout))
                    return tin, tout, ttot
                out0 = as_dict.get("output") or [{}]
                if isinstance(out0, list) and out0:
                    u = out0[0].get("usage", {}) or {}
                    tin = int(u.get("input_tokens", 0))
                    tout = int(u.get("output_tokens", 0))
                    ttot = int(u.get("total_tokens", tin + tout))
                    return tin, tout, ttot
        except Exception:
            pass
        return 0, 0, 0

    def _accumulate_and_print(
        self, label: str, name: str, t0: float, resp: Any
    ) -> None:
        tin, tout, ttot = self._usage_from_resp(resp)
        sec = time.time() - t0
        tqdm.write(
            f"[{label}] {name} | {sec:.2f}s | tokens in/out/total: {tin}/{tout}/{ttot}"
        )
        self._n += 1
        self._sum_sec += sec
        self._sum_in += tin
        self._sum_out += tout
        self._sum_tot += ttot

    def create_batch_requests(
        self, input_dir: Path, first_n: Optional[int] = None
    ) -> List[Dict]:
        """Create Batch request objects for all files in directory"""
        batch_requests = []

        pdf_files = list(input_dir.glob("*.pdf"))
        # jsonl_files = list(input_dir.glob("*.jsonl"))

        if first_n:
            pdf_files = pdf_files[:first_n]

        # Process PDFs
        for idx, pdf_path in enumerate(pdf_files):
            out_dir = SETTINGS.paths.output_dir / pdf_path.stem
            if out_dir.exists():
                continue

            try:
                file_id = self.upload_pdf_get_id(pdf_path)
                custom_id = f"pdf_{pdf_path.stem}_{idx}"

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": MODEL,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_file", "file_id": file_id},
                                    {"type": "input_text", "text": PROMPT_EXTRACT},
                                ],
                            }
                        ],
                        "reasoning": {"effort": REASONING_EFFORT},
                    },
                }
                batch_requests.append(
                    {
                        "request": request,
                        "file_path": pdf_path,
                        "file_type": "pdf",
                        "meta": [{"key": "filename", "value": pdf_path.name}],
                    }
                )
            except Exception as e:
                write_failure(SETTINGS.paths.ouput_dir, pdf_path.stem, e)

        # Process JSONL files
        json_items_cap = first_n if first_n else None
        # TEMPORARY: Only process arxiv.jsonl
        arxiv_jsonl_path = input_dir / "arxiv.jsonl"
        jsonl_files_filtered = [arxiv_jsonl_path] if arxiv_jsonl_path.exists() else []

        # for jsonl_path in jsonl_files:
        for jsonl_path in jsonl_files_filtered:
            if json_items_cap is not None and json_items_cap <= 0:
                break

            try:
                with jsonl_path.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if (
                            json_items_cap is not None
                            and len(batch_requests) >= first_n
                        ):
                            break

                        line = line.strip()
                        if not line:
                            continue

                        try:
                            paper_json = json.loads(line)
                        except Exception as e:
                            pid = f"{jsonl_path.stem}__badjson_{idx}"
                            write_failure(SETTINGS.paths.output_dir, pid, e)
                            continue

                        paper_id = (
                            jsonl_path.stem
                            + "__"
                            + url_to_id(paper_json.get("url", f"line_{idx}"))
                        )
                        out_dir = SETTINGS.paths.output_dir / paper_id
                        if out_dir.exists():
                            continue

                        custom_id = f"jsonl_{paper_id}_{idx}"

                        request = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/responses",
                            "body": {
                                "model": MODEL,
                                "input": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_text",
                                                "text": PROMPT_EXTRACT,
                                            },
                                            {
                                                "type": "input_text",
                                                "text": f"\n\nHere is the paper for analysis:\n\n{paper_json['text']}",
                                            },
                                        ],
                                    }
                                ],
                                "reasoning": {"effort": REASONING_EFFORT},
                            },
                        }

                        batch_requests.append(
                            {
                                "request": request,
                                "paper_id": paper_id,
                                "file_type": "jsonl",
                                "meta": filter_dict(paper_json, META_KEYS),
                            }
                        )

            except Exception as e:
                write_failure(SETTINGS.paths.output_dir, jsonl_path.stem, e)

        return batch_requests

    def create_batch_input_file(self, batch_requests: List[Dict]) -> str:
        """Create and upload the batch input JSONL file."""
        batch_input_path = SETTINGS.paths.output_dir / "batch_input.jsonl"

        with batch_input_path.open("w", encoding="utf-8") as f:
            for batch_req in batch_requests:
                f.write(json.dumps(batch_req["request"], ensure_ascii=False) + "\n")

        # Upload batch input file
        with batch_input_path.open("rb") as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch",
            )

        print(f"Uploaded batch input file: {batch_input_file.id}")
        return batch_input_file.id

    def create_batch_job(
        self, input_file_id: str, description: str = "Paper extraction batch"
    ) -> str:
        """Create batch job for paper extraction."""
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": description},
        )

        print(f"Created batch job: {batch.id}")
        print(f"Status: {batch.status}")
        return batch.id

    def wait_for_batch_completion(
        self, batch_id: str, check_interval: int = 60
    ) -> Dict:
        """Wait for batch to complete, checking status periodically."""
        print(f"Waiting for batch {batch_id} to complete...")

        while True:
            batch = self.client.batches.retrieve(batch_id)
            print(f"Batch status: {batch.status}")

            if batch.status in ["completed", "failed", "expired", "cancelled"]:
                return batch
            if batch.status == "failed":
                raise RuntimeError(f"Batch failed: {batch}")

            time.sleep(check_interval)

    def process_batch_results(self, batch: Dict, batch_requests: List[Dict]) -> None:
        """Process and save batch results."""
        if batch.status != "completed":
            print(f"Batch did not complete successfully. Status: {batch.status}")
            return

        # Create mapping of custom_id to request metadata
        custom_id_map = {req["request"]["custom_id"]: req for req in batch_requests}

        # Download and process results
        if batch.output_file_id:
            file_response = self.client.files.content(batch.output_file_id)
            results_content = file_response.text

            total_tokens = 0
            processed_count = 0

            for line in results_content.strip().split("\n"):
                if not line:
                    continue

                try:
                    result = json.loads(line)
                    custom_id = result["custom_id"]

                    if custom_id not in custom_id_map:
                        print(f"Warning: Unknown custom_id {custom_id}")
                        continue

                    req_data = custom_id_map[custom_id]

                    if result["error"]:
                        print(f"Error for {custom_id}: {result['error']}")
                        continue

                    # Extract response data
                    response_body = result["response"]["body"]

                    # Determine output directory and stem
                    if req_data["file_type"] == "pdf":
                        file_path = req_data["file_path"]
                        out_dir = SETTINGS.paths.output_dir / file_path.stem
                        stem = file_path.stem
                    else:  # jsonl
                        paper_id = req_data["paper_id"]
                        out_dir = SETTINGS.paths.output_dir / paper_id
                        stem = paper_id

                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Save outputs (adapt the response format as needed)
                    self.write_batch_outputs(
                        out_dir, stem, response_body, req_data["meta"]
                    )

                    # Track usage
                    if "usage" in response_body:
                        total_tokens += response_body["usage"].get("total_tokens", 0)

                    processed_count += 1

                except Exception as e:
                    print(f"Error processing result line: {e}")
                    print(f"Line: {line}")

            print("\n=== Batch Processing Summary ===")
            print(f"Papers processed: {processed_count}")
            print(f"Total tokens used: {total_tokens}")

        # Process error file if it exists
        if batch.error_file_id:
            error_response = self.client.files.content(batch.error_file_id)
            error_content = error_response.text
            print("\nError occured during batch processing:")
            for line in error_content.strip().split("\n"):
                if line:
                    error_result = json.loads(line)
                    print(
                        f"Error for {error_result['custom_id']}: {error_result['error']}"
                    )

    def write_batch_outputs(
        self, out_dir: Path, stem: str, response_body: Dict, meta: Dict
    ) -> None:
        """Write batch response outputs to correspondant files."""
        raw_path = out_dir / f"{stem}_raw_response.txt"
        json_path = out_dir / f"{stem}.json"
        summary_path = out_dir / f"{stem}_summary.txt"

        # Save raw response
        safe_write(raw_path, json.dumps(response_body, ensure_ascii=False, indent=2))

        # Extract output text from response
        # Note: You may need to adjust this based on the actual response format
        output_text = ""
        if "choices" in response_body and response_body["choices"]:
            choice = response_body["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                output_text = choice["message"]["content"]

        if output_text:
            text_part, json_part = split_text_and_json(output_text)
            safe_write(summary_path, text_part or "")

            if json_part:
                try:
                    parsed = json.loads(json_part)
                    if meta:
                        parsed["meta"] = meta
                    safe_write(
                        json_path, json.dumps(parsed, ensure_ascii=False, indent=2)
                    )
                except json.JSONDecodeError as e:
                    write_failure(
                        SETTINGS.paths.output_dir, stem, f"JSON parse error: {e}"
                    )

    def process_dir_batch(
        self,
        input_dir: Path,
        first_n: Optional[int] = None,
        description: str = "Paper extraction batch",
    ) -> None:
        """Process directory using Batch API."""
        SETTINGS.paths.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating batch requests for files in {input_dir}")
        batch_requests = self.create_batch_requests(input_dir, first_n)

        if not batch_requests:
            print("No files to process.")
            return

        print(f"Created {len(batch_requests)} batch requests")

        # Split general batch into smaller batches if needed (max 50k requests per batch)
        max_batch_size = 50000

        for i in range(0, len(batch_requests), max_batch_size):
            batch_chunk = batch_requests[i : i + max_batch_size]
            batch_num = i // max_batch_size + 1

            print(
                f"\n=== Processing Batch {batch_num} ({len(batch_chunk)} requests) ==="
            )

            # Create and upload batch input file
            input_file_id = self.create_batch_input_file(batch_chunk)

            # Create batch job
            batch_description = f"{description} - Batch {batch_num}"
            batch_id = self.create_batch_job(input_file_id, batch_description)

            # Wait for completion
            completed_batch = self.wait_for_batch_completion(batch_id)

            # Process results
            self.process_batch_results(completed_batch, batch_chunk)

    def process_pdf(self, path: Path) -> None:
        out_dir = SETTINGS.paths.output_dir / path.stem
        if out_dir.exists():
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        file_id = self.upload_pdf_get_id(path)
        resp = self.call_openai_file(file_id)
        meta = [{"key": "filename", "value": path.name}]
        self.write_outputs(out_dir, path.stem, resp, meta)
        self._accumulate_and_print("PDF", path.name, t0, resp)

    def process_jsonl(self, path: Path, max_items: Optional[int] = None) -> int:
        processed = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                pbar = tqdm(total=max_items, desc=f"{path.name} (items)", leave=False)
                for idx, line in enumerate(f, start=1):
                    if max_items is not None and processed >= max_items:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        paper_json = json.loads(line)
                    except Exception as e:
                        pid = f"{path.stem}__badjson_{idx}"
                        write_failure(SETTINGS.paths.output_dir, pid, e)
                        processed += 1
                        pbar.update(1)
                        continue

                    paper_id = (
                        path.stem
                        + "__"
                        + url_to_id(paper_json.get("url", f"line_{idx}"))
                    )
                    out_dir = SETTINGS.paths.output_dir / paper_id
                    if out_dir.exists():
                        processed += 1
                        pbar.update(1)
                        continue

                    try:
                        t0 = time.time()
                        resp = self.call_openai_text(paper_json["text"])
                        meta = filter_dict(paper_json, META_KEYS)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        self.write_outputs(out_dir, paper_id, resp, meta)
                        self._accumulate_and_print("JSONL", paper_id, t0, resp)
                    except Exception as e:
                        write_failure(SETTINGS.paths.output_dir, paper_id, e)
                    finally:
                        processed += 1
                        pbar.update(1)
                pbar.close()
        except Exception as e:
            raise RuntimeError(f"Failed to read JSON file '{path}': {e}") from e
        return processed

    def process_dir(self, input_dir: Path, first_n: Optional[int] = None) -> None:
        SETTINGS.paths.output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_dir.glob("*.pdf"))
        jsonl_files = list(input_dir.glob("*.jsonl"))

        if first_n:
            pdf_files = pdf_files[:first_n]

        print(
            f"Found {len(pdf_files)} PDFs and {len(jsonl_files)} JSONLs in {input_dir} to process."
        )
        print(pdf_files + jsonl_files)

        for file in tqdm(pdf_files, desc="PDFs"):
            try:
                self.process_pdf(file)
            except Exception as e:
                write_failure(SETTINGS.paths.output_dir, file.stem, e)

        json_items_cap = first_n if first_n else None
        if jsonl_files and (json_items_cap is None or json_items_cap > 0):
            for file in tqdm(jsonl_files, desc="JSONL files"):
                try:
                    taken = self.process_jsonl(file, max_items=json_items_cap)
                    if json_items_cap is not None:
                        json_items_cap -= taken
                        if json_items_cap <= 0:
                            break
                except Exception as e:
                    write_failure(SETTINGS.paths.output_dir, file.stem, e)

        if self._n:
            avg_sec = self._sum_sec / self._n
            avg_tok = self._sum_tot / self._n
            print("\n=== Summary ===")
            print(f"Papers processed:     {self._n}")
            print(f"Total time (sec):     {self._sum_sec:.2f}")
            print(f"Avg time/paper (sec): {avg_sec:.2f}")
            print(
                f"Total tokens:         {self._sum_tot} (in={self._sum_in}, out={self._sum_out})"
            )
            print(f"Avg tokens/paper:     {int(avg_tok)}")
        else:
            print("\nNo papers processed.")


if __name__ == "__main__":
    extractor = Extractor()
    # extractor.process_dir(SETTINGS.paths.input_dir, 200)
    extractor.process_dir_batch(SETTINGS.paths.input_dir, 3)
