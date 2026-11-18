import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI
from pydantic import BaseModel, Field

from app.rag import search_patches
from app.sql_tool import run_readonly_sql

MODEL_NAME = "gemma-4-E4B-it"
MAX_STEPS = 5


class SearchPatchesArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Search keywords for VALORANT Patch 9.08 notes or tactical agent guides.")
    min_score: float = Field(default=0.015, ge=0.0, le=1.0, description="Minimum RRF score threshold.")


class QueryValorantDbArgs(BaseModel):
    sql: str = Field(..., min_length=6, description="Single read-only SQLite SELECT statement querying weapons or abilities.")


class CoachStructuredResponse(BaseModel):
    answer: str
    citations: List[str]
    tool_trace: List[Dict[str, Any]]
    steps_used: int
    model: str


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_patches",
            "description": (
                "Hybrid BM25 + Vector RRF retrieval over VALORANT Patch 9.08 notes (patch_9_08.md) "
                "and tactical agent & post-plant guides (agent_guides.md). Returns ranked chunks with [source_filename] citations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query about VALORANT patch changes, map updates, agent tactics, or post-plant economy.",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum RRF fusion score threshold (default 0.015).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_valorant_db",
            "description": (
                "Execute a read-only SQLite SELECT query on data/valorant.db. "
                "Tables: weapons(id, name, category, cost, fire_rate, magazine_size, wall_penetration, head_damage_close, body_damage_close, leg_damage_close) "
                "and abilities(id, agent, role, slot, name, cost_or_ult_points, max_charges, description)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single valid SQLite SELECT statement without comments or stacked queries.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are the VALORANT Tactical & Patch Coach, an expert assistant built for competitive VALORANT players and analysts.
Follow these strict rules:
1. For any question about Patch 9.08 balance changes, Sunset map updates, competitive map pool rotation, agent playbooks (Jett, Omen, Sova, Cypher), or post-plant/economy doctrine, call `search_patches`.
2. For any question about weapon stats (cost, fire_rate, magazine_size, wall_penetration, head_damage_close, body_damage_close, leg_damage_close) or ability stats/costs (agent, role, slot, name, cost_or_ult_points, max_charges, description), call `query_valorant_db` with a clean SQLite SELECT query.
3. For hybrid questions requiring both patch/tactical guidance and weapon/ability numbers, call both tools before answering.
4. Always ground your answer in tool results and cite retrieved markdown sources inline using exact bracketed filenames such as [patch_9_08.md] or [agent_guides.md], and cite [valorant.db] when reporting SQL database statistics.
5. Never invent stats or patch changes not present in the tool outputs. If a tool returns no matching data, state that honestly."""


def get_llm_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable. Set OPENAI_API_KEY before calling the LLM proxy.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return client


def extract_citations(text: str, tool_trace: List[Dict[str, Any]]) -> List[str]:
    found = set(re.findall(r"\[([a-zA-Z0-9_.-]+\.(?:md|db))\]", text))
    for item in tool_trace:
        if item.get("tool") == "search_patches" and isinstance(item.get("output"), dict):
            for hit in item["output"].get("results", []):
                src = hit.get("source")
                if src:
                    found.add(src)
        elif item.get("tool") == "query_valorant_db" and isinstance(item.get("output"), dict):
            if item["output"].get("ok") and item["output"].get("row_count", 0) > 0:
                found.add("valorant.db")
    return sorted(found)


def run_agent(user_message: str, max_steps: int = MAX_STEPS) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    tool_trace: List[Dict[str, Any]] = []
    seen_calls: set[str] = set()
    steps_used = 0

    client = get_llm_client()

    for step in range(1, max_steps + 1):
        steps_used = step
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            final_text = msg.content or ""
            citations = extract_citations(final_text, tool_trace)
            for c in citations:
                tag = f"[{c}]"
                if tag not in final_text:
                    final_text = f"{final_text.rstrip()} {tag}"
            validated = CoachStructuredResponse(
                answer=final_text.strip(),
                citations=citations,
                tool_trace=tool_trace,
                steps_used=steps_used,
                model=MODEL_NAME,
            )
            return validated.model_dump()

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            fn_name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = {}

            normalized_sig = f"{fn_name}::{json.dumps(parsed_args, sort_keys=True)}"
            if normalized_sig in seen_calls:
                dup_output = {
                    "warning": "Duplicate tool call blocked by loop guard. Use existing observations to synthesize the final answer."
                }
                tool_trace.append(
                    {
                        "step": step,
                        "tool": fn_name,
                        "input": parsed_args,
                        "duplicate_blocked": True,
                        "output": dup_output,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": json.dumps(dup_output),
                    }
                )
                continue

            seen_calls.add(normalized_sig)

            if fn_name == "search_patches":
                try:
                    valid_args = SearchPatchesArgs(**parsed_args)
                    tool_result = search_patches(query=valid_args.query, min_score=valid_args.min_score)
                except Exception as exc:
                    tool_result = {"error": f"Invalid arguments for search_patches: {exc}"}
            elif fn_name == "query_valorant_db":
                try:
                    valid_args = QueryValorantDbArgs(**parsed_args)
                    tool_result = run_readonly_sql(sql=valid_args.sql)
                except Exception as exc:
                    tool_result = {"ok": False, "error": f"Invalid arguments for query_valorant_db: {exc}"}
            else:
                tool_result = {"error": f"Tool '{fn_name}' is not on the allow-list."}

            tool_trace.append(
                {
                    "step": step,
                    "tool": fn_name,
                    "input": parsed_args,
                    "duplicate_blocked": False,
                    "output": tool_result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

    fallback_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages + [{"role": "user", "content": "Provide your final grounded answer with [source] citations now."}],
        temperature=0,
    )
    final_text = fallback_resp.choices[0].message.content or "Reached max agent steps."
    citations = extract_citations(final_text, tool_trace)
    for c in citations:
        tag = f"[{c}]"
        if tag not in final_text:
            final_text = f"{final_text.rstrip()} {tag}"
    validated = CoachStructuredResponse(
        answer=final_text.strip(),
        citations=citations,
        tool_trace=tool_trace,
        steps_used=steps_used,
        model=MODEL_NAME,
    )
    return validated.model_dump()
