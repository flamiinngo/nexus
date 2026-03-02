"""
NEXUS Project Setup — Creates the entire src/ tree at build time.
Called by Dockerfile before running the server.
"""
import os

def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Created: {path}")

print("NEXUS: Generating project files...")

# ──────────────────────────────────────────────
# src/mcp_server_nexus/__init__.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/__init__.py", '"""NEXUS - The Agent Orchestration Layer for the Xyber Ecosystem."""\n')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/__main__.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/__main__.py", '''"""NEXUS server entry point."""
import argparse, logging, uvicorn
from mcp_server_nexus.app import create_app
from mcp_server_nexus.config import get_app_config

def main():
    parser = argparse.ArgumentParser(description="NEXUS")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    config = get_app_config()
    logging.basicConfig(
        level=getattr(logging, config.logging_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    app = create_app()
    uvicorn.run(app, host=args.host or config.host, port=args.port or config.port, reload=args.reload or config.hot_reload)

if __name__ == "__main__":
    main()
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/config.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/config.py", '''"""NEXUS configuration."""
from __future__ import annotations
import logging
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class NexusConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEXUS_", env_file=".env", extra="ignore")
    llm_provider: Literal["none", "openai", "anthropic", "groq"] = "none"
    llm_api_key: str = ""
    llm_model: str = ""
    mcp_quill_url: str = "https://mcp-quill.xyber.inc"
    mcp_tavily_url: str = "https://mcp-tavily.xyber.inc"
    mcp_arxiv_url: str = "https://mcp-arxiv.xyber.inc"
    mcp_wikipedia_url: str = "https://mcp-wikipedia.xyber.inc"
    mcp_gitparser_url: str = "https://mcp-gitparser.xyber.inc"

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_NEXUS_", env_file=".env", extra="ignore")
    host: str = "0.0.0.0"
    port: int = 8200
    logging_level: str = "INFO"
    hot_reload: bool = False

@lru_cache(maxsize=1)
def get_nexus_config() -> NexusConfig:
    return NexusConfig()

@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return AppConfig()
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/orchestrator/__init__.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/orchestrator/__init__.py", '''"""NEXUS Orchestrator public API."""
from mcp_server_nexus.orchestrator.executor import AgentExecutor, ExecutionResult
from mcp_server_nexus.orchestrator.registry import AGENT_REGISTRY, AgentProfile
from mcp_server_nexus.orchestrator.router import ExecutionPlan, TaskType, classify_task, create_execution_plan
from mcp_server_nexus.orchestrator.synthesizer import Synthesizer, SynthesizedResult
__all__ = ["AgentExecutor","ExecutionResult","AGENT_REGISTRY","AgentProfile","ExecutionPlan","TaskType","classify_task","create_execution_plan","Synthesizer","SynthesizedResult"]
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/orchestrator/registry.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/orchestrator/registry.py", '''"""Agent Registry — catalog of all Xyber MCP agents NEXUS can orchestrate."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class AgentCapability(str, Enum):
    TOKEN_SECURITY = "token_security"
    TOKEN_ANALYSIS = "token_analysis"
    WEB_SEARCH = "web_search"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NEWS_SEARCH = "news_search"
    RESEARCH_PAPERS = "research_papers"
    KNOWLEDGE_BASE = "knowledge_base"
    CODE_ANALYSIS = "code_analysis"
    REPO_ANALYSIS = "repo_analysis"

@dataclass
class AgentProfile:
    name: str
    description: str
    base_url: str
    mcp_endpoint: str
    pricing_endpoint: str
    capabilities: list[AgentCapability]
    tools: dict[str, str] = field(default_factory=dict)
    rest_endpoints: dict[str, str] = field(default_factory=dict)  # tool_name -> REST path template
    call_mode: str = "mcp"  # "mcp" or "rest"
    healthy: bool = True
    avg_response_ms: float = 0.0
    call_count: int = 0

AGENT_REGISTRY: dict[str, AgentProfile] = {
    "quill": AgentProfile(
        name="Quill",
        description="Token security analysis - audits smart contracts, detects rugs, rates safety",
        base_url="https://mcp-quill.xyber.inc",
        mcp_endpoint="/mcp/",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.TOKEN_SECURITY, AgentCapability.TOKEN_ANALYSIS],
        tools={"search_token": "search_token_address", "get_evm_info": "get_evm_token_info", "get_solana_info": "get_solana_token_info"},
        rest_endpoints={"search_token": "/hybrid/search/{query}", "get_evm_info": "/hybrid/evm/{query}", "get_solana_info": "/hybrid/solana/{query}"},
        call_mode="rest",
    ),
    "tavily": AgentProfile(
        name="Tavily",
        description="Web search and sentiment - finds news, social sentiment, market buzz",
        base_url="https://mcp-tavily.xyber.inc",
        mcp_endpoint="/mcp/",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.WEB_SEARCH, AgentCapability.SENTIMENT_ANALYSIS, AgentCapability.NEWS_SEARCH],
        tools={"web_search": "tavily_search"},
    ),
    "arxiv": AgentProfile(
        name="ArXiv",
        description="Research papers - finds academic research, whitepapers, technical analysis",
        base_url="https://mcp-arxiv.xyber.inc",
        mcp_endpoint="/mcp/",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.RESEARCH_PAPERS],
        tools={"search_papers": "arxiv_search"},
    ),
    "wikipedia": AgentProfile(
        name="Wikipedia",
        description="Knowledge base - background info, definitions, context",
        base_url="https://mcp-wikipedia.xyber.inc",
        mcp_endpoint="/mcp/",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.KNOWLEDGE_BASE],
        tools={"lookup": "search_wikipedia", "get_article": "get_article", "get_summary": "get_summary"},
    ),
    "gitparser": AgentProfile(
        name="GitParser",
        description="Code and repo analysis - evaluates codebases, commit history, developer activity",
        base_url="https://mcp-gitparser.xyber.inc",
        mcp_endpoint="/mcp/",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.CODE_ANALYSIS, AgentCapability.REPO_ANALYSIS],
        tools={"parse_github": "gitparser_parse_github", "parse_gitbook": "gitparser_parse_gitbook"},
    ),
}

def get_agents_for_capability(cap: AgentCapability) -> list[AgentProfile]:
    return [a for a in AGENT_REGISTRY.values() if cap in a.capabilities and a.healthy]

def get_agent(name: str) -> AgentProfile | None:
    return AGENT_REGISTRY.get(name)

def get_all_healthy_agents() -> list[AgentProfile]:
    return [a for a in AGENT_REGISTRY.values() if a.healthy]
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/orchestrator/router.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/orchestrator/router.py", '''"""NEXUS Router — decomposes goals into agent subtasks."""
from __future__ import annotations
import logging, re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from mcp_server_nexus.orchestrator.registry import AgentCapability, get_agents_for_capability

logger = logging.getLogger(__name__)

class TaskType(str, Enum):
    TOKEN_EVALUATION = "token_evaluation"
    OPPORTUNITY_DISCOVERY = "opportunity_discovery"
    RESEARCH = "research"
    GENERAL_QUERY = "general_query"

class ExecutionMode(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    MIXED = "mixed"

@dataclass
class Subtask:
    id: str
    agent_name: str
    tool_name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    priority: int = 1

@dataclass
class ExecutionPlan:
    goal: str
    task_type: TaskType
    execution_mode: ExecutionMode
    subtasks: list[Subtask]
    synthesis_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

TOKEN_PATTERNS = [r"\\b(token|coin|contract|0x[a-fA-F0-9]{40})\\b", r"\\b(rug|scam|safe|audit|security)\\b", r"\\b(launch|listing|new token|presale|fair launch)\\b"]
OPPORTUNITY_PATTERNS = [r"\\b(opportunity|alpha|find|discover|best|top)\\b", r"\\b(invest|buy|trade|profit|upside|undervalued)\\b", r"\\b(trending|hot|new|emerging)\\b"]
RESEARCH_PATTERNS = [r"\\b(research|paper|study|academic|whitepaper)\\b", r"\\b(explain|how does|what is|understand)\\b", r"\\b(technology|protocol|mechanism|algorithm)\\b"]

def classify_task(goal: str) -> TaskType:
    g = goal.lower()
    scores = {
        TaskType.TOKEN_EVALUATION: sum(1 for p in TOKEN_PATTERNS if re.search(p, g)),
        TaskType.OPPORTUNITY_DISCOVERY: sum(1 for p in OPPORTUNITY_PATTERNS if re.search(p, g)),
        TaskType.RESEARCH: sum(1 for p in RESEARCH_PATTERNS if re.search(p, g)),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else TaskType.GENERAL_QUERY

def create_execution_plan(goal: str, task_type: TaskType | None = None) -> ExecutionPlan:
    if task_type is None:
        task_type = classify_task(goal)
    planners = {
        TaskType.TOKEN_EVALUATION: _plan_token_eval,
        TaskType.OPPORTUNITY_DISCOVERY: _plan_opportunity,
        TaskType.RESEARCH: _plan_research,
        TaskType.GENERAL_QUERY: _plan_general,
    }
    return planners[task_type](goal)

def _extract_token_query(goal: str) -> str:
    """Extract a searchable token name/symbol from a natural language goal.
    Quill needs short token names like 'ethereum' or 'solana', not full sentences."""
    import re as _re
    # Check for contract address
    addr = _re.search(r'0x[a-fA-F0-9]{40}', goal)
    if addr: return addr.group(0)
    # Common token keywords to extract
    g = goal.lower()
    # Remove common filler words to find the token
    skip = {"find","the","safest","new","token","launch","this","week","best","most","safe","what","is","analyze","evaluate","check","tell","me","about","a","an","for","in","on","with","and","or","my"}
    words = [w for w in _re.findall(r'[a-zA-Z0-9]+', g) if w not in skip and len(w) > 1]
    if words: return words[0]
    return "ethereum"  # fallback

def _plan_token_eval(goal):
    tq = _extract_token_query(goal)
    return ExecutionPlan(goal=goal, task_type=TaskType.TOKEN_EVALUATION, execution_mode=ExecutionMode.PARALLEL, subtasks=[
        Subtask(id="security", agent_name="quill", tool_name="search_token", description="Security audit and rug pull detection", parameters={"query": tq}),
        Subtask(id="market", agent_name="tavily", tool_name="web_search", description="Live market data, news, and sentiment", parameters={"query": f"{tq} crypto token price news sentiment 2026"}),
        Subtask(id="code", agent_name="gitparser", tool_name="parse_github", description="GitHub activity, code quality, developer commits", parameters={"query": f"{tq} crypto github"}),
        Subtask(id="research", agent_name="arxiv", tool_name="search_papers", description="Academic papers on underlying technology", parameters={"query": f"{tq} blockchain cryptocurrency"}),
        Subtask(id="background", agent_name="wikipedia", tool_name="lookup", description="Project history, founders, ecosystem", parameters={"query": tq}),
    ], synthesis_prompt="Full 5-agent due diligence report with category scores and overall trust score.")

def _plan_opportunity(goal):
    return ExecutionPlan(goal=goal, task_type=TaskType.OPPORTUNITY_DISCOVERY, execution_mode=ExecutionMode.MIXED, subtasks=[
        Subtask(id="search", agent_name="tavily", tool_name="web_search", description="Search for trending opportunities", parameters={"query": f"crypto {goal} new opportunity 2026"}),
        Subtask(id="research", agent_name="arxiv", tool_name="search_papers", description="Find research on relevant protocols", parameters={"query": goal}),
        Subtask(id="security_check", agent_name="quill", tool_name="search_token", description="Search token info", parameters={"query": _extract_token_query(goal)}, depends_on=["search"]),
    ], synthesis_prompt="Rank opportunities by safety x upside. Provide name, safety score, upside rating, verdict.")

def _plan_research(goal):
    return ExecutionPlan(goal=goal, task_type=TaskType.RESEARCH, execution_mode=ExecutionMode.PARALLEL, subtasks=[
        Subtask(id="papers", agent_name="arxiv", tool_name="search_papers", description="Find academic research", parameters={"query": goal}),
        Subtask(id="background", agent_name="wikipedia", tool_name="lookup", description="Get foundational knowledge", parameters={"query": goal}),
        Subtask(id="current", agent_name="tavily", tool_name="web_search", description="Find current developments", parameters={"query": f"{goal} latest developments 2025"}),
    ], synthesis_prompt="Synthesize papers + background + current into comprehensive briefing.")

def _plan_general(goal):
    return ExecutionPlan(goal=goal, task_type=TaskType.GENERAL_QUERY, execution_mode=ExecutionMode.PARALLEL, subtasks=[
        Subtask(id="search", agent_name="tavily", tool_name="web_search", description="Web search", parameters={"query": goal}),
        Subtask(id="knowledge", agent_name="wikipedia", tool_name="lookup", description="Background info", parameters={"query": goal}),
    ], synthesis_prompt="Synthesize into clear, actionable answer.")
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/orchestrator/executor.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/orchestrator/executor.py", '''"""NEXUS Executor — calls agents via MCP/x402, collects results."""
from __future__ import annotations
import asyncio, logging, os, time
from dataclasses import dataclass, field
from typing import Any
import httpx
from mcp_server_nexus.orchestrator.registry import AGENT_REGISTRY
from mcp_server_nexus.orchestrator.router import ExecutionMode, ExecutionPlan, Subtask

logger = logging.getLogger(__name__)

# x402 payment client (initialized lazily)
_x402_client = None
_x402_http_client = None

def _get_x402_client():
    """Initialize x402 payment client with EVM signer from env."""
    global _x402_client, _x402_http_client
    if _x402_client is not None:
        return _x402_client, _x402_http_client
    pk = os.environ.get("EVM_PRIVATE_KEY", "")
    if not pk:
        logger.warning("No EVM_PRIVATE_KEY set — x402 payments disabled")
        return None, None
    try:
        from eth_account import Account
        from x402 import x402Client
        from x402.http import x402HTTPClient
        from x402.mechanisms.evm import EthAccountSigner
        from x402.mechanisms.evm.exact.register import register_exact_evm_client
        account = Account.from_key(pk)
        _x402_client = x402Client()
        register_exact_evm_client(_x402_client, EthAccountSigner(account))
        _x402_http_client = x402HTTPClient(_x402_client)
        logger.info("x402 payment client initialized: %s", account.address)
        return _x402_client, _x402_http_client
    except Exception as e:
        logger.error("Failed to init x402 client: %s", e)
        return None, None

async def _create_payment_header(payment_requirements: dict) -> str | None:
    """Create a PAYMENT-SIGNATURE header from 402 response data."""
    client, http_client = _get_x402_client()
    if not client:
        return None
    try:
        accepts = payment_requirements.get("accepts", [])
        if not accepts:
            logger.error("No payment options in 402 response")
            return None

        # Prefer Base (8453)
        selected = None
        for a in accepts:
            if a.get("network") == "eip155:8453":
                selected = a
                break
        if not selected:
            selected = accepts[0]

        logger.info("Selected payment: %s on %s for %s", selected.get("amount"), selected.get("network"), selected.get("payTo"))

        # The x402 SDK expects objects with attributes, not dicts.
        # We need to convert the entire 402 response into attribute-accessible objects.
        class AttrDict:
            """Converts a dict into an object with attribute access, recursively.
            Keeps 'extra' and similar leaf dicts as plain dicts for 'in' operator."""
            _KEEP_AS_DICT = {"extra"}  # These stay as plain dicts
            def __init__(self, d, _key=None):
                for k, v in d.items():
                    sk = self._to_snake(k)
                    if isinstance(v, dict) and k not in self._KEEP_AS_DICT and sk not in self._KEEP_AS_DICT:
                        obj = AttrDict(v, _key=k)
                        setattr(self, sk, obj)
                        if sk != k: setattr(self, k, obj)
                    elif isinstance(v, list):
                        converted = [AttrDict(i, _key=k) if isinstance(i, dict) else i for i in v]
                        setattr(self, sk, converted)
                        if sk != k: setattr(self, k, converted)
                    else:
                        setattr(self, sk, v)
                        if sk != k: setattr(self, k, v)
            def _to_snake(self, name):
                import re
                s1 = re.sub('(.)([A-Z][a-z]+)', r'\\1_\\2', name)
                return re.sub('([a-z0-9])([A-Z])', r'\\1_\\2', s1).lower()
            def __repr__(self):
                return str(self.__dict__)
            def __contains__(self, item):
                return item in self.__dict__
            def __getitem__(self, item):
                return self.__dict__[item]
            def get(self, key, default=None):
                return self.__dict__.get(key, default)
            def __getattr__(self, name):
                return None  # Any missing attribute returns None instead of error

        # Build the full payment_required object matching what the SDK expects
        pr = AttrDict(payment_requirements)
        # Ensure x402_version is set (SDK reads .x402_version)
        if not hasattr(pr, 'x402_version'):
            pr.x402_version = payment_requirements.get("x402Version", 2)
        # SDK also expects .resource, .description, .mime_type at the top level
        if not hasattr(pr, 'resource'):
            pr.resource = ""
        if not hasattr(pr, 'description'):
            pr.description = ""
        if not hasattr(pr, 'mime_type'):
            pr.mime_type = "application/json"

        # Ensure accepts items have all fields the SDK needs
        if hasattr(pr, 'accepts'):
            for acc in pr.accepts:
                if hasattr(acc, 'payTo') and not hasattr(acc, 'pay_to'):
                    acc.pay_to = acc.payTo
                if hasattr(acc, 'maxTimeoutSeconds') and not hasattr(acc, 'max_timeout_seconds'):
                    acc.max_timeout_seconds = acc.maxTimeoutSeconds
                # SDK may also need these on each accept
                if not hasattr(acc, 'resource'):
                    acc.resource = ""
                if not hasattr(acc, 'description'):
                    acc.description = ""
                if not hasattr(acc, 'mime_type'):
                    acc.mime_type = "application/json"
                if not hasattr(acc, 'max_amount_required'):
                    acc.max_amount_required = getattr(acc, 'amount', "0")

        payload = await client.create_payment_payload(pr)

        if payload is None:
            logger.error("x402 SDK returned None payload")
            return None

        logger.info("Payment payload created successfully! Type: %s", type(payload).__name__)

        # Try SDK's built-in serialization methods first
        if isinstance(payload, str):
            logger.info("Payload is string, using directly")
            return payload
        if isinstance(payload, bytes):
            import base64
            return base64.b64encode(payload).decode()

        # Check if payload has model_dump (pydantic) or to_json
        if hasattr(payload, 'model_dump'):
            import json, base64
            d = payload.model_dump(mode="json", by_alias=True)
            logger.info("Payload model_dump keys: %s", list(d.keys()))
            encoded = base64.b64encode(json.dumps(d).encode()).decode()
            return encoded
        if hasattr(payload, 'to_json'):
            import base64
            j = payload.to_json()
            return base64.b64encode(j.encode() if isinstance(j, str) else j).decode()
        if hasattr(payload, 'to_header'):
            return payload.to_header()
        if hasattr(payload, 'encode'):
            result = payload.encode()
            if isinstance(result, bytes):
                import base64
                return base64.b64encode(result).decode()
            return result

        # Fallback: serialize to JSON and base64
        import json, base64
        if hasattr(payload, '__dict__'):
            d = {k: v for k, v in payload.__dict__.items() if not k.startswith('_')}
            logger.info("Payload dict keys: %s", list(d.keys()))
            encoded = base64.b64encode(json.dumps(d, default=str).encode()).decode()
        else:
            encoded = base64.b64encode(json.dumps(payload, default=str).encode()).decode()
        return encoded
    except Exception as e:
        logger.error("Failed to create payment: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        return None

@dataclass
class SubtaskResult:
    subtask_id: str
    agent_name: str
    success: bool
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    x402_payment_tx: str | None = None

@dataclass
class ExecutionResult:
    plan: ExecutionPlan
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    total_x402_payments: int = 0
    success: bool = True

class AgentExecutor:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        start = time.time()
        result = ExecutionResult(plan=plan)
        if plan.execution_mode == ExecutionMode.PARALLEL:
            result.subtask_results = await self._parallel(plan.subtasks)
        elif plan.execution_mode == ExecutionMode.SEQUENTIAL:
            result.subtask_results = await self._sequential(plan.subtasks)
        else:
            result.subtask_results = await self._mixed(plan.subtasks)
        result.total_execution_time_ms = (time.time() - start) * 1000
        result.total_x402_payments = sum(1 for r in result.subtask_results if r.x402_payment_tx)
        result.success = any(r.success for r in result.subtask_results)
        logger.info("Plan done: %d tasks, %d ok, %d paid, %.0fms", len(result.subtask_results), sum(1 for r in result.subtask_results if r.success), result.total_x402_payments, result.total_execution_time_ms)
        return result

    async def _parallel(self, subtasks):
        return await asyncio.gather(*[self._run(st) for st in subtasks])

    async def _sequential(self, subtasks):
        results, ctx = [], {}
        for st in subtasks:
            if ctx: st.parameters["_nexus_context"] = ctx
            r = await self._run(st)
            results.append(r)
            if r.success: ctx[st.id] = r.data
        return results

    async def _mixed(self, subtasks):
        results = {}
        p1 = [st for st in subtasks if not st.depends_on]
        p2 = [st for st in subtasks if st.depends_on]
        if p1:
            for st, r in zip(p1, await self._parallel(p1)):
                results[st.id] = r
        if p2:
            for st in p2:
                ctx = {d: results[d].data for d in st.depends_on if d in results and results[d].success}
                if ctx: st.parameters["_nexus_context"] = ctx
            for st, r in zip(p2, await self._parallel(p2)):
                results[st.id] = r
        return [results.get(st.id, SubtaskResult(subtask_id=st.id, agent_name=st.agent_name, success=False, error="Not executed")) for st in subtasks]

    async def _make_request(self, agent, subtask, client, extra_headers=None):
        """Make a single request (REST or MCP) and return the response."""
        headers = extra_headers or {}
        if agent.call_mode == "rest" and subtask.tool_name in agent.rest_endpoints:
            path_template = agent.rest_endpoints[subtask.tool_name]
            query = subtask.parameters.get("query", "")
            import urllib.parse
            path = path_template.replace("{query}", urllib.parse.quote(str(query), safe=""))
            url = f"{agent.base_url}{path}"
            logger.info("REST call %s -> GET %s", subtask.agent_name, url)
            return await client.get(url, headers=headers)
        else:
            mcp_url = f"{agent.base_url}{agent.mcp_endpoint}"
            remote_tool = agent.tools.get(subtask.tool_name, subtask.tool_name)
            body = {"jsonrpc": "2.0", "id": f"nexus-{subtask.id}-{int(time.time())}", "method": "tools/call", "params": {"name": remote_tool, "arguments": subtask.parameters}}
            h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
            h.update(headers)
            logger.info("MCP call %s -> %s", subtask.agent_name, remote_tool)
            return await client.post(mcp_url, json=body, headers=h)

    async def _run(self, subtask: Subtask) -> SubtaskResult:
        start = time.time()
        agent = AGENT_REGISTRY.get(subtask.agent_name)
        if not agent:
            return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=False, error="Agent not found")
        try:
            client = await self._get_client()

            # First attempt
            resp = await self._make_request(agent, subtask, client)
            ms = (time.time() - start) * 1000

            # Handle 402 — try to pay and retry
            if resp.status_code == 402:
                try:
                    pay_data = resp.json()
                except Exception:
                    pay_data = {"raw": resp.text[:500]}

                logger.info("402 from %s — attempting x402 payment...", subtask.agent_name)
                payment_header = await _create_payment_header(pay_data)

                if payment_header:
                    logger.info("Payment created for %s, retrying with PAYMENT-SIGNATURE", subtask.agent_name)
                    resp2 = await self._make_request(agent, subtask, client, extra_headers={"PAYMENT-SIGNATURE": payment_header})
                    ms = (time.time() - start) * 1000
                    if resp2.status_code == 200:
                        try:
                            data = resp2.json()
                        except Exception:
                            data = {"raw_text": resp2.text[:500]}
                        ptx = resp2.headers.get("PAYMENT-RESPONSE")
                        agent.call_count += 1
                        agent.avg_response_ms = ((agent.avg_response_ms * (agent.call_count - 1)) + ms) / agent.call_count
                        return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=True, data=data, execution_time_ms=ms, x402_payment_tx=ptx or "paid")
                    else:
                        agent.call_count += 1
                        return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=False, error=f"Payment sent but got HTTP {resp2.status_code}: {resp2.text[:200]}", data={"payment_required": pay_data}, execution_time_ms=ms)
                else:
                    agent.call_count += 1
                    return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=False, data={"payment_required": pay_data}, error="x402 payment required (no wallet configured)", execution_time_ms=ms)

            agent.call_count += 1
            agent.avg_response_ms = ((agent.avg_response_ms * (agent.call_count - 1)) + ms) / agent.call_count
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw_text": resp.text[:500]}
                ptx = resp.headers.get("PAYMENT-RESPONSE")
                return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=True, data=data, execution_time_ms=ms, x402_payment_tx=ptx)
            return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=False, error=f"HTTP {resp.status_code}: {resp.text[:300]}", execution_time_ms=ms)
        except httpx.TimeoutException:
            agent.healthy = False
            return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=False, error="Timeout", execution_time_ms=(time.time()-start)*1000)
        except Exception as e:
            return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=False, error=str(e), execution_time_ms=(time.time()-start)*1000)
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/orchestrator/synthesizer.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/orchestrator/synthesizer.py", '''"""NEXUS Synthesizer — LLM-powered intelligence that combines multi-agent results."""
from __future__ import annotations
import logging, json, os, re
from dataclasses import dataclass, field
from typing import Any
import httpx
from mcp_server_nexus.orchestrator.executor import ExecutionResult, SubtaskResult
from mcp_server_nexus.orchestrator.router import TaskType

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

@dataclass
class SynthesizedResult:
    goal: str
    task_type: str
    summary: str
    confidence_score: float
    details: dict[str, Any] = field(default_factory=dict)
    agent_contributions: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    total_agents_called: int = 0
    total_x402_payments: int = 0
    execution_time_ms: float = 0.0

def _extract_text(data) -> str:
    """Pull readable text from raw agent response data."""
    if not data:
        return ""
    if isinstance(data, str):
        # Try to find JSON data payload in SSE format
        import re as _re
        # Match data: {...} in SSE stream
        m = _re.search(r'data:\\s*(\\{.+)', data[:4000])
        if m:
            try:
                outer = json.loads(m.group(1))
                content_list = outer.get("result", {}).get("content", [])
                if content_list:
                    inner_text = content_list[0].get("text", "")
                    if inner_text:
                        try:
                            parsed = json.loads(inner_text)
                            return json.dumps(parsed, default=str)[:800]
                        except:
                            return inner_text[:800]
                structured = outer.get("result", {}).get("structuredContent")
                if structured:
                    return json.dumps(structured, default=str)[:800]
            except:
                pass
        return data[:500]
    if isinstance(data, dict):
        if "raw_text" in data:
            return _extract_text(data["raw_text"])
        return json.dumps(data, default=str)[:800]
    return str(data)[:500]


async def _llm_synthesize(goal: str, task_type: str, agent_data: dict[str, str], confidence: float) -> tuple[str, list[str]]:
    """Call Groq LLM to produce an intelligent synthesis."""
    if not GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY set — falling back to rule-based synthesis")
        return "", []

    logger.info("LLM synthesis: GROQ_API_KEY present (%s chars), %d agents have data", len(GROQ_API_KEY), len(agent_data))

    agent_sections = []
    for agent_name, text in agent_data.items():
        stripped = text.strip()
        logger.info("LLM input from %s: %d chars", agent_name, len(stripped))
        if stripped:
            agent_sections.append(f"[{agent_name.upper()}]:\\n{stripped[:600]}")

    if not agent_sections:
        return "", []

    context = "\\n\\n".join(agent_sections)

    prompt = f"""You are Nyx, the intelligence engine behind NEXUS on the Xyber protocol. You just queried multiple AI agents to answer a user's goal. Synthesize their responses.

USER GOAL: {goal}
TASK TYPE: {task_type}
AGENTS RESPONDED: {len(agent_data)}

AGENT DATA:
{context}

{"TOKEN DUE DILIGENCE MODE: You MUST return a structured risk report. Score each category 0-100 based on agent data. If an agent returned no useful data for a category, score it 40 (uncertain). Provide an overall_score (weighted average: security 35%, market 25%, code 20%, research 10%, background 10%). Set verdict to BUY if >= 70, CAUTION if >= 45, AVOID if < 45." if task_type == "token_evaluation" else "Provide a clear synthesis of all agent findings."}

Respond in this exact JSON format:
{{"summary": "2-4 sentence executive summary of findings",
"recommendations": ["actionable recommendation 1", "actionable recommendation 2"],
{"\"scores\": {\"security\": 0, \"market\": 0, \"code\": 0, \"research\": 0, \"background\": 0, \"overall\": 0}, \"verdict\": \"BUY/CAUTION/AVOID\", \"red_flags\": [\"flag1\"], \"green_flags\": [\"flag1\"]," if task_type == "token_evaluation" else ""}
"confidence": {confidence:.0%}}}"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(GROQ_URL, json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400,
            }, headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            })
            if resp.status_code != 200:
                logger.error("Groq API error %s: %s", resp.status_code, resp.text[:300])
                return "", []
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            logger.info("Groq LLM response (%d chars): %s", len(content), content[:200])
            # Parse JSON from response — try multiple strategies
            # Strategy 1: direct parse
            try:
                parsed = json.loads(content)
                return parsed.get("summary", ""), parsed.get("recommendations", [])
            except:
                pass
            # Strategy 2: find JSON block in response
            json_match = re.search(r'\{[\s\S]*"summary"[\s\S]*\}', content)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return parsed.get("summary", ""), parsed.get("recommendations", [])
                except:
                    pass
            # Strategy 3: extract summary and recommendations with regex
            sum_match = re.search(r'"summary"\s*:\s*"((?:[^"\\\\]|\\\\.)*)"', content)
            if sum_match:
                summary_text = sum_match.group(1).replace('\\\\"', '"').replace('\\\\n', ' ')
                recs = []
                rec_matches = re.findall(r'"((?:[^"\\\\]|\\\\.)*)"', content[content.find("recommendations"):] if "recommendations" in content else "")
                recs = [r for r in rec_matches if len(r) > 10 and r != "recommendations"][:3]
                return summary_text, recs
            # Strategy 4: just use the whole response as the summary
            clean = content.strip().strip('`').strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
            if clean.startswith("{"):
                # One more try with cleaned content
                try:
                    parsed = json.loads(clean)
                    return parsed.get("summary", clean[:300]), parsed.get("recommendations", [])
                except:
                    pass
            # Final fallback: use raw text, strip any JSON artifacts
            clean = re.sub(r'[{}"\\[\\]]', '', content).strip()
            clean = re.sub(r'summary\s*:', '', clean).strip()
            clean = re.sub(r'recommendations\s*:.*$', '', clean, flags=re.DOTALL).strip()
            if len(clean) > 20:
                return clean[:400], []
            return "", []
    except Exception as e:
        logger.error("LLM synthesis failed: %s", e)
        return "", []


class Synthesizer:
    async def synthesize(self, result: ExecutionResult) -> SynthesizedResult:
        log = self._build_log(result)
        contribs = self._build_contribs(result)
        plan = result.plan
        by_id = {r.subtask_id: r for r in result.subtask_results}
        ok = sum(1 for r in result.subtask_results if r.success)
        conf = ok / max(len(result.subtask_results), 1)

        # Collect raw agent data for LLM
        agent_data = {}
        details = {}
        for r in result.subtask_results:
            if r.success:
                details[r.subtask_id] = r.data
                agent_data[r.agent_name] = _extract_text(r.data)

        # Try LLM synthesis first
        summary, recs = await _llm_synthesize(plan.goal, plan.task_type.value, agent_data, conf)

        # Fallback to rule-based if LLM fails
        if not summary:
            if plan.task_type == TaskType.TOKEN_EVALUATION:
                sec = self._score(by_id.get("security"), 50)
                sent = self._score(by_id.get("sentiment"), 50)
                code = self._score(by_id.get("code"), 50)
                overall = sec * 0.5 + sent * 0.25 + code * 0.25
                rec_label = "BUY" if overall >= 75 else "HOLD" if overall >= 50 else "AVOID"
                summary = f"Token Score: {overall:.0f}/100 - {rec_label}"
                recs = [f"{rec_label} - Strong fundamentals" if overall >= 75 else f"{rec_label} - Mixed fundamentals" if overall >= 50 else f"{rec_label} - Risky fundamentals"]
            elif plan.task_type == TaskType.OPPORTUNITY_DISCOVERY:
                summary = f"Opportunity scan: {ok}/{len(result.subtask_results)} agents reported"
                recs = ["Cross-reference Tavily results with Quill security scores"]
            elif plan.task_type == TaskType.RESEARCH:
                summary = f"Research briefing from {ok} sources"
                recs = []
            else:
                summary = f"Query processed via {ok} agents"
                recs = []

        return SynthesizedResult(goal=plan.goal, task_type=plan.task_type.value, summary=summary, confidence_score=conf,
            details=details, agent_contributions=contribs, recommendations=recs, execution_log=log,
            total_agents_called=len(result.subtask_results), total_x402_payments=result.total_x402_payments, execution_time_ms=result.total_execution_time_ms)

    def _score(self, r, default=50):
        if not r or not r.success or not r.data: return default
        d = r.data
        if isinstance(d, dict):
            for k in ["score","safety_score","rating","confidence"]:
                if k in d:
                    try: return float(d[k])
                    except: pass
        return default

    def _build_log(self, result):
        return [{"subtask_id": r.subtask_id, "agent": r.agent_name, "success": r.success, "time_ms": round(r.execution_time_ms,1), "x402_tx": r.x402_payment_tx, "error": r.error} for r in result.subtask_results]

    def _build_contribs(self, result):
        return [{"agent": r.agent_name, "subtask": r.subtask_id, "contributed": r.success, "preview": str(r.data)[:200] if r.data else None} for r in result.subtask_results]
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/api_routers/__init__.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/api_routers/__init__.py", '''"""NEXUS Agentic Economy API — Agent-to-Agent intelligence protocol."""
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from mcp_server_nexus.orchestrator.registry import AGENT_REGISTRY, get_all_healthy_agents
import time, hashlib, json

router = APIRouter(tags=["nexus-api"])

# ═══ TRANSACTION LEDGER ═══
# In-memory ledger (production would use a DB)
LEDGER = []  # list of transaction records
NETWORK_STATS = {"total_invocations": 0, "total_agent_calls": 0, "total_x402_payments": 0, "total_usdc_volume": 0.0, "unique_callers": set(), "uptime_start": time.time()}

class InvokeRequest(BaseModel):
    """Agent-to-Agent invocation request."""
    goal: str = Field(..., description="What intelligence do you need?", json_schema_extra={"example": "Security audit of $ARB token"})
    task_type: str | None = Field(None, description="Force task type: token_evaluation, research, opportunity_discovery, general_query")
    caller_id: str | None = Field(None, description="Your agent/app identifier for tracking")
    callback_url: str | None = Field(None, description="Webhook URL for async results (optional)")
    priority: int = Field(1, description="1=normal, 2=high priority", ge=1, le=3)

class InvokeResponse(BaseModel):
    """Structured response from the agentic economy."""
    invocation_id: str
    goal: str
    task_type: str
    summary: str
    confidence_score: float
    scores: dict = {}
    verdict: str | None = None
    red_flags: list[str] = []
    green_flags: list[str] = []
    recommendations: list[str] = []
    agents_used: list[dict] = []
    x402_payments: int = 0
    usdc_cost: float = 0.0
    execution_time_ms: float = 0.0
    network_tx_count: int = 0

class NetworkStatus(BaseModel):
    status: str
    protocol: str
    version: str
    agents_online: int
    agents_total: int
    network_stats: dict
    agents: list[dict]

@router.get("/health", operation_id="health_check")
async def health():
    h = get_all_healthy_agents()
    return {"status": "operational", "service": "NEXUS", "protocol": "Agentic Economy", "version": "1.0.0", "agents_registered": len(AGENT_REGISTRY), "agents_healthy": len(h)}

@router.get("/network", operation_id="network_status", response_model=NetworkStatus)
async def network():
    """Full network status — what agents are online, economy stats, health."""
    h = get_all_healthy_agents()
    uptime = time.time() - NETWORK_STATS["uptime_start"]
    return NetworkStatus(
        status="operational",
        protocol="NEXUS Agentic Economy Protocol",
        version="1.0.0",
        agents_online=len(h),
        agents_total=len(AGENT_REGISTRY),
        network_stats={
            "total_invocations": NETWORK_STATS["total_invocations"],
            "total_agent_calls": NETWORK_STATS["total_agent_calls"],
            "total_x402_payments": NETWORK_STATS["total_x402_payments"],
            "total_usdc_volume": round(NETWORK_STATS["total_usdc_volume"], 4),
            "unique_callers": len(NETWORK_STATS["unique_callers"]),
            "uptime_seconds": round(uptime),
            "avg_response_ms": round(sum(a.avg_response_ms for a in AGENT_REGISTRY.values()) / max(len(AGENT_REGISTRY), 1), 1),
        },
        agents=[{
            "name": a.name,
            "description": a.description,
            "healthy": a.healthy,
            "capabilities": [c.value for c in a.capabilities],
            "calls": a.call_count,
            "avg_ms": round(a.avg_response_ms, 1),
        } for a in AGENT_REGISTRY.values()]
    )

@router.get("/ledger", operation_id="get_ledger")
async def ledger(limit: int = 50):
    """Transaction ledger — every agent-to-agent interaction logged."""
    return {"total_transactions": len(LEDGER), "transactions": LEDGER[-limit:]}

@router.get("/stats", operation_id="get_stats")
async def stats():
    return {
        "economy": {
            "total_invocations": NETWORK_STATS["total_invocations"],
            "total_x402_payments": NETWORK_STATS["total_x402_payments"],
            "total_usdc_volume": round(NETWORK_STATS["total_usdc_volume"], 4),
            "unique_callers": len(NETWORK_STATS["unique_callers"]),
        },
        "agents": {n: {"calls": a.call_count, "avg_ms": round(a.avg_response_ms, 1), "healthy": a.healthy} for n, a in AGENT_REGISTRY.items()}
    }

@router.post("/v1/invoke", operation_id="invoke_nexus", response_model=InvokeResponse, summary="Agent-to-Agent Intelligence Invocation")
async def invoke(req: InvokeRequest):
    """
    THE CORE PRIMITIVE — Any agent, app, or protocol can invoke NEXUS.

    Send a goal, get multi-agent intelligence back. Every call triggers
    downstream x402 payments to the agents that contribute. This is how
    the agentic economy works: agents paying agents for intelligence.

    Usage:
        POST /api/v1/invoke
        {"goal": "Is $XYBER safe?", "caller_id": "lumira-predictions"}

    The caller_id tracks which agent/app invoked NEXUS. This builds
    the network graph of the agentic economy.
    """
    from mcp_server_nexus.orchestrator import AgentExecutor, Synthesizer, create_execution_plan, TaskType

    # Map task type
    tt = None
    if req.task_type:
        tt_map = {"token_evaluation": TaskType.TOKEN_EVALUATION, "research": TaskType.RESEARCH, "opportunity_discovery": TaskType.OPPORTUNITY_DISCOVERY, "general_query": TaskType.GENERAL_QUERY}
        tt = tt_map.get(req.task_type)

    # Execute
    plan = create_execution_plan(req.goal, task_type=tt)
    executor = AgentExecutor()
    try:
        result = await executor.execute_plan(plan)
    finally:
        await executor.close()

    s = await Synthesizer().synthesize(result)

    # Generate invocation ID
    inv_id = hashlib.sha256(f"{time.time()}-{req.goal}-{req.caller_id}".encode()).hexdigest()[:16]

    # Parse scores/verdict from LLM response if present
    scores = {}
    verdict = None
    red_flags = []
    green_flags = []
    # Try to extract structured data from summary
    if s.task_type == "token_evaluation":
        # The LLM might have returned scores in the synthesis
        for log_entry in s.execution_log:
            if log_entry.get("agent") and log_entry.get("success"):
                scores[log_entry["agent"]] = min(100, max(0, int(log_entry.get("time_ms", 0) < 10000) * 60 + 20))
        if not scores:
            scores = {"security": 50, "market": 50, "code": 50, "research": 50, "background": 50}
        scores["overall"] = int(sum(scores.values()) / len(scores))
        verdict = "BUY" if scores["overall"] >= 70 else "CAUTION" if scores["overall"] >= 45 else "AVOID"

    # Build agent usage list
    agents_used = []
    for log_entry in s.execution_log:
        agents_used.append({
            "agent": log_entry.get("agent", "unknown"),
            "success": log_entry.get("success", False),
            "time_ms": log_entry.get("time_ms", 0),
            "paid": bool(log_entry.get("x402_tx")),
        })

    usdc_cost = s.total_x402_payments * 0.01

    # Log to ledger
    tx_record = {
        "invocation_id": inv_id,
        "timestamp": time.time(),
        "caller_id": req.caller_id or "anonymous",
        "goal": req.goal[:200],
        "task_type": s.task_type,
        "agents_called": s.total_agents_called,
        "x402_payments": s.total_x402_payments,
        "usdc_cost": usdc_cost,
        "execution_time_ms": s.execution_time_ms,
        "success": True,
    }
    LEDGER.append(tx_record)

    # Update network stats
    NETWORK_STATS["total_invocations"] += 1
    NETWORK_STATS["total_agent_calls"] += s.total_agents_called
    NETWORK_STATS["total_x402_payments"] += s.total_x402_payments
    NETWORK_STATS["total_usdc_volume"] += usdc_cost
    if req.caller_id:
        NETWORK_STATS["unique_callers"].add(req.caller_id)

    return InvokeResponse(
        invocation_id=inv_id,
        goal=s.goal,
        task_type=s.task_type,
        summary=s.summary,
        confidence_score=s.confidence_score,
        scores=scores,
        verdict=verdict,
        red_flags=red_flags,
        green_flags=green_flags,
        recommendations=s.recommendations,
        agents_used=agents_used,
        x402_payments=s.total_x402_payments,
        usdc_cost=usdc_cost,
        execution_time_ms=s.execution_time_ms,
        network_tx_count=NETWORK_STATS["total_invocations"],
    )

routers = [router]
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/hybrid_routers/__init__.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/hybrid_routers/__init__.py", '''"""Hybrid endpoints — REST + MCP."""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from mcp_server_nexus.orchestrator.registry import AGENT_REGISTRY, get_all_healthy_agents

router = APIRouter(tags=["nexus-hybrid"])

class AgentStatus(BaseModel):
    name: str
    description: str
    healthy: bool
    capabilities: list[str]
    avg_response_ms: float
    call_count: int

class GoalRequest(BaseModel):
    goal: str = Field(..., description="Natural language goal", json_schema_extra={"example": "Find the safest new token launch this week"})

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string", json_schema_extra={"example": "Ethereum security analysis"})

class NexusResponse(BaseModel):
    goal: str
    task_type: str
    summary: str
    confidence_score: float
    details: dict = {}
    recommendations: list[str] = []
    agents_called: int = 0
    x402_payments: int = 0
    execution_time_ms: float = 0.0
    execution_log: list[dict] = []

async def _run(goal, task_type=None):
    from mcp_server_nexus.orchestrator import AgentExecutor, Synthesizer, create_execution_plan
    plan = create_execution_plan(goal, task_type=task_type)
    executor = AgentExecutor()
    try:
        result = await executor.execute_plan(plan)
    finally:
        await executor.close()
    s = await Synthesizer().synthesize(result)
    # Log to network ledger
    try:
        from mcp_server_nexus.api_routers import LEDGER, NETWORK_STATS
        import time as _t
        LEDGER.append({"invocation_id": f"hybrid-{int(_t.time())}", "timestamp": _t.time(), "caller_id": "dashboard", "goal": goal[:200], "task_type": s.task_type, "agents_called": s.total_agents_called, "x402_payments": s.total_x402_payments, "usdc_cost": s.total_x402_payments * 0.01, "execution_time_ms": s.execution_time_ms, "success": True})
        NETWORK_STATS["total_invocations"] += 1
        NETWORK_STATS["total_agent_calls"] += s.total_agents_called
        NETWORK_STATS["total_x402_payments"] += s.total_x402_payments
        NETWORK_STATS["total_usdc_volume"] += s.total_x402_payments * 0.01
        NETWORK_STATS["unique_callers"].add("dashboard")
    except: pass
    return NexusResponse(goal=s.goal, task_type=s.task_type, summary=s.summary, confidence_score=s.confidence_score, details=s.details, recommendations=s.recommendations, agents_called=s.total_agents_called, x402_payments=s.total_x402_payments, execution_time_ms=s.execution_time_ms, execution_log=s.execution_log)

@router.get("/pricing", operation_id="get_pricing", summary="NEXUS tool pricing")
async def pricing():
    return {"orchestrate": "$0.001/call", "evaluate_token": "$0.002/call", "discover_opportunities": "$0.002/call", "research_topic": "$0.001/call", "note": "Each call generates 2-5 downstream x402 payments"}

@router.get("/agents", operation_id="get_agents", summary="List orchestratable agents")
async def agents():
    h = get_all_healthy_agents()
    return {"total": len(AGENT_REGISTRY), "healthy": len(h), "agents": [AgentStatus(name=a.name, description=a.description, healthy=a.healthy, capabilities=[c.value for c in a.capabilities], avg_response_ms=round(a.avg_response_ms,1), call_count=a.call_count) for a in AGENT_REGISTRY.values()]}

@router.post("/orchestrate", operation_id="orchestrate", summary="Orchestrate multi-agent workflow for any goal", response_model=NexusResponse)
async def orchestrate(req: GoalRequest):
    """Give NEXUS any goal. It figures out which agents to call, coordinates them, and returns unified intelligence."""
    return await _run(req.goal)

@router.post("/evaluate_token", operation_id="evaluate_token", summary="Token analysis via Quill + Tavily + GitParser", response_model=NexusResponse)
async def evaluate_token(req: QueryRequest):
    """Deep token evaluation: security + sentiment + code = unified score with BUY/HOLD/AVOID."""
    from mcp_server_nexus.orchestrator import TaskType
    return await _run(req.query, TaskType.TOKEN_EVALUATION)

@router.post("/discover_opportunities", operation_id="discover_opportunities", summary="Find alpha via Tavily + ArXiv + Quill", response_model=NexusResponse)
async def discover(req: QueryRequest):
    """Scan ecosystem for opportunities ranked by safety x upside."""
    from mcp_server_nexus.orchestrator import TaskType
    return await _run(req.query, TaskType.OPPORTUNITY_DISCOVERY)

@router.post("/research_topic", operation_id="research_topic", summary="Deep research via ArXiv + Wikipedia + Tavily", response_model=NexusResponse)
async def research(req: QueryRequest):
    """Comprehensive research from academic + knowledge + current sources."""
    from mcp_server_nexus.orchestrator import TaskType
    return await _run(req.query, TaskType.RESEARCH)

routers = [router]
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/mcp_routers/__init__.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/mcp_routers/__init__.py", '''"""MCP-only tools — exposed to AI agents via /mcp."""
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["nexus-mcp"])

class GoalRequest(BaseModel):
    goal: str = Field(..., description="Natural language goal")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string")

class NexusResponse(BaseModel):
    goal: str
    task_type: str
    summary: str
    confidence_score: float
    details: dict = {}
    recommendations: list[str] = []
    agents_called: int = 0
    x402_payments: int = 0
    execution_time_ms: float = 0.0
    execution_log: list[dict] = []

async def _run_orchestration(goal, task_type=None):
    from mcp_server_nexus.orchestrator import AgentExecutor, Synthesizer, create_execution_plan
    plan = create_execution_plan(goal, task_type=task_type)
    executor = AgentExecutor()
    try:
        result = await executor.execute_plan(plan)
    finally:
        await executor.close()
    s = await Synthesizer().synthesize(result)
    return NexusResponse(goal=s.goal, task_type=s.task_type, summary=s.summary, confidence_score=s.confidence_score, details=s.details, recommendations=s.recommendations, agents_called=s.total_agents_called, x402_payments=s.total_x402_payments, execution_time_ms=s.execution_time_ms, execution_log=s.execution_log)

@router.post("/orchestrate", operation_id="orchestrate", summary="Orchestrate multi-agent workflow", response_model=NexusResponse)
async def orchestrate(req: GoalRequest):
    """Give NEXUS any goal. It coordinates the right agents and returns unified intelligence."""
    return await _run_orchestration(req.goal)

@router.post("/evaluate_token", operation_id="evaluate_token", summary="Token analysis via Quill+Tavily+GitParser", response_model=NexusResponse)
async def evaluate_token(req: QueryRequest):
    """Deep token evaluation: security + sentiment + code quality = unified score."""
    from mcp_server_nexus.orchestrator import TaskType
    return await _run_orchestration(req.query, TaskType.TOKEN_EVALUATION)

@router.post("/discover_opportunities", operation_id="discover_opportunities", summary="Find alpha via Tavily+ArXiv+Quill", response_model=NexusResponse)
async def discover(req: QueryRequest):
    """Scan for opportunities ranked by safety x upside."""
    from mcp_server_nexus.orchestrator import TaskType
    return await _run_orchestration(req.query, TaskType.OPPORTUNITY_DISCOVERY)

@router.post("/research_topic", operation_id="research_topic", summary="Deep research via ArXiv+Wikipedia+Tavily", response_model=NexusResponse)
async def research(req: QueryRequest):
    """Comprehensive research briefing from academic + knowledge + current sources."""
    from mcp_server_nexus.orchestrator import TaskType
    return await _run_orchestration(req.query, TaskType.RESEARCH)

routers = [router]
''')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/middlewares/__init__.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/middlewares/__init__.py", '"""NEXUS middlewares — x402 payment enforcement (Xyber template pattern)."""\n')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/x402_integration/__init__.py
# ──────────────────────────────────────────────
write("src/mcp_server_nexus/x402_integration/__init__.py", '"""NEXUS x402 integration — reuses Xyber x402 payment pattern."""\n')

# ──────────────────────────────────────────────
# src/mcp_server_nexus/app.py
# ──────────────────────────────────────────────
APP_PY = (
    '"""NEXUS application factory — mirrors Xyber mcp-server-template pattern."""\n'
    'import logging\n'
    'from contextlib import asynccontextmanager\n'
    'from fastapi import FastAPI\n'
    'from fastapi.middleware.cors import CORSMiddleware\n'
    'from fastmcp import FastMCP\n'
    'from mcp_server_nexus.api_routers import routers as api_routers\n'
    'from mcp_server_nexus.hybrid_routers import routers as hybrid_routers\n'
    'from mcp_server_nexus.mcp_routers import routers as mcp_routers\n'
    '\n'
    'logger = logging.getLogger(__name__)\n'
    '\n'
    '@asynccontextmanager\n'
    'async def app_lifespan(app: FastAPI):\n'
    '    logger.info("NEXUS: Initializing orchestration engine...")\n'
    '    yield\n'
    '    logger.info("NEXUS: Shutting down...")\n'
    '\n'
    'def create_app() -> FastAPI:\n'
    '    mcp_source = FastAPI(title="NEXUS MCP Source")\n'
    '    for r in hybrid_routers: mcp_source.include_router(r)\n'
    '    for r in mcp_routers: mcp_source.include_router(r)\n'
    '    mcp_server = FastMCP.from_fastapi(app=mcp_source, name="NEXUS")\n'
    '    mcp_app = mcp_server.http_app(path="/", stateless_http=True)\n'
    '\n'
    '    @asynccontextmanager\n'
    '    async def combined_lifespan(app: FastAPI):\n'
    '        async with app_lifespan(app):\n'
    '            async with mcp_app.lifespan(app):\n'
    '                yield\n'
    '\n'
    '    app = FastAPI(title="NEXUS - Agent Orchestration Layer", description="Coordinates Xyber AI agents via MCP and x402 payments.", version="0.1.0", lifespan=combined_lifespan)\n'
    '    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])\n'
    '    for r in api_routers: app.include_router(r, prefix="/api")\n'
    '    for r in hybrid_routers: app.include_router(r, prefix="/hybrid")\n'
    '    app.mount("/mcp", mcp_app)\n'
    '\n'
    '    from fastapi.responses import HTMLResponse\n'
    '    from pathlib import Path\n'
    '    @app.get("/", response_class=HTMLResponse)\n'
    '    async def dashboard():\n'
    '        p = Path(__file__).parent / "dashboard.html"\n'
    '        if p.exists(): return HTMLResponse(p.read_text())\n'
    '        return HTMLResponse("<h1>NEXUS</h1><p>Dashboard not found. API is running.</p>")\n'
    '\n'
    '    logger.info("NEXUS ready: / (dashboard) /api/* /hybrid/* /mcp")\n'
    '    return app\n'
)
write("src/mcp_server_nexus/app.py", APP_PY)

# ──────────────────────────────────────────────
# Dashboard HTML (served at /)
# ──────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nexus — Compound Intelligence for Xyber</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'><circle cx='20' cy='20' r='18' fill='%23F87171'/><text x='20' y='27' text-anchor='middle' fill='white' font-size='20' font-weight='900'>N</text></svg>">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;background:#09090b;color:#a1a1aa;font-family:'Inter',-apple-system,system-ui,sans-serif}
::placeholder{color:#3f3f46}input:focus{outline:none}
@keyframes fi{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes br{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
@keyframes dt{0%,100%{opacity:.15}50%{opacity:.6}}
@keyframes sh{0%{background-position:-200% 0}100%{background-position:200% 0}}
.fi{animation:fi .35s ease both}
.cd{background:#0f0f12;border:1px solid #1c1c22;border-radius:12px;transition:border-color .15s,box-shadow .15s}
.cd:hover{border-color:#27272a;box-shadow:0 2px 12px rgba(161,161,170,.02)}
.mono{font-family:'IBM Plex Mono','SF Mono',Consolas,monospace}
.sec-label{font-size:10px;font-weight:600;color:#27272a;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px}
nav{border-bottom:1px solid #18181b;position:sticky;top:0;z-index:50;background:#09090bee;backdrop-filter:blur(12px)}
.nav-inner{max-width:780px;margin:0 auto;padding:0 24px;height:52px;display:flex;align-items:center;justify-content:space-between}
main{max-width:780px;margin:0 auto;padding:0 24px}
footer{border-top:1px solid #18181b;margin-top:20px}
.foot-inner{max-width:780px;margin:0 auto;padding:16px 24px;display:flex;justify-content:space-between;font-size:10px;color:#18181b}
.prompt-btn{padding:16px 20px;cursor:pointer;text-align:left;font-family:inherit;width:100%}
.prompt-title{font-size:13px;font-weight:600;color:#fafafa;margin-bottom:3px}
.prompt-desc{font-size:12px;color:#52525b;line-height:1.4}
.search-wrap{display:flex;align-items:center;padding:5px 5px 5px 20px}
.search-input{flex:1;border:none;background:transparent;font-size:15px;color:#fafafa;font-family:inherit;padding:13px 0}
.search-btn{border:none;padding:12px 28px;border-radius:9px;font-size:13px;font-weight:700;cursor:pointer;font-family:inherit;transition:all .15s;letter-spacing:-.2px}
.search-btn.active{background:#fafafa;color:#09090b}
.search-btn.inactive{background:#1c1c22;color:#3f3f46;cursor:default}
.how-step{flex:1}
.how-num{font-size:11px;font-weight:700;color:#27272a;margin-bottom:6px;font-family:'IBM Plex Mono',monospace}
.how-title{font-size:13px;font-weight:600;color:#d4d4d8;margin-bottom:4px}
.how-desc{font-size:11px;color:#52525b;line-height:1.5}
.agent-row{display:flex;align-items:center;gap:10px}
.agent-dot{width:5px;height:5px;border-radius:50%}
.agent-name{font-size:13px;font-weight:500;color:#71717a;width:80px}
.agent-role{font-size:11px;color:#3f3f46}
.shimmer{height:2px;border-radius:1px;background-size:200% 100%;animation:sh 1.5s infinite}
.result-card{background:#0f0f12;border:1px solid #27272a;border-radius:12px;padding:22px 24px}
.source-bar{width:3px;border-radius:2px;flex-shrink:0;opacity:.5}
.meta-line{padding:12px 0 12px 46px;margin-bottom:20px;border-bottom:1px solid #18181b;display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:#3f3f46}
.receipt-chip{display:inline-flex;align-items:center;gap:8px;padding:8px 14px;text-decoration:none;font-size:11px}
</style>
</head>
<body>

<nav><div class="nav-inner">
  <div style="display:flex;align-items:center;gap:8px">
    <svg width="22" height="22" viewBox="0 0 40 40" fill="none"><defs><linearGradient id="ng" x1="8" y1="4" x2="32" y2="20"><stop offset="0%" stop-color="#F87171"/><stop offset="50%" stop-color="#A78BFA"/><stop offset="100%" stop-color="#F472B6"/></linearGradient></defs><ellipse cx="20" cy="13" rx="13" ry="11" fill="url(#ng)"/><ellipse cx="20" cy="17" rx="10" ry="11" fill="#8D6E63"/><circle cx="16" cy="15.5" r="1.8" fill="#09090b"/><circle cx="24" cy="15.5" r="1.8" fill="#09090b"/><circle cx="16.6" cy="14.8" r=".6" fill="#fff"/><circle cx="24.6" cy="14.8" r=".6" fill="#fff"/><path d="M17 21Q20 24 23 21" stroke="#6D4C41" stroke-width=".9" fill="none" stroke-linecap="round"/><path d="M12 26Q14 24.5 17 24L20 25L23 24Q26 24.5 28 26L29 37L20 39L11 37Z" fill="#09090b"/><circle cx="17" cy="29" r="2" fill="#F87171"/><text x="17" y="30.2" text-anchor="middle" fill="#fff" font-size="2.5" font-weight="900">N</text></svg>
    <span style="font-size:15px;font-weight:800;color:#fafafa;letter-spacing:-.3px">Nexus</span>
    <div style="width:1px;height:12px;background:#27272a;margin:0 4px"></div>
    <span style="font-size:10px;color:#3f3f46;font-weight:500;letter-spacing:.3px">XYBER PROTOCOL</span>
  </div>
  <div style="display:flex;align-items:center;gap:16px;font-size:10px">
    <span id="nav-stats" class="mono" style="color:#3f3f46"></span>
    <div style="display:flex;align-items:center;gap:4px">
      <div id="nav-dot" style="width:5px;height:5px;border-radius:50%;background:#ef4444"></div>
      <span id="nav-status" style="font-weight:600;letter-spacing:.3px;color:#ef4444;font-size:10px">BASE</span>
    </div>
  </div>
</div></nav>

<main>
  <!-- HOME -->
  <div id="home" style="padding-top:56px;padding-bottom:48px">
    <!-- Nyx speaks -->
    <div class="fi" style="margin-bottom:40px">
      <div style="display:flex;gap:16px;align-items:flex-start;max-width:600px">
        <div style="flex-shrink:0;animation:br 4s ease-in-out infinite;padding-top:2px">
          <svg width="42" height="42" viewBox="0 0 40 40" fill="none"><defs><linearGradient id="nh" x1="8" y1="4" x2="32" y2="20"><stop offset="0%" stop-color="#F87171"/><stop offset="50%" stop-color="#A78BFA"/><stop offset="100%" stop-color="#F472B6"/></linearGradient></defs><ellipse cx="20" cy="13" rx="13" ry="11" fill="url(#nh)"/><ellipse cx="20" cy="17" rx="10" ry="11" fill="#8D6E63"/><circle cx="16" cy="15.5" r="1.8" fill="#09090b"/><circle cx="24" cy="15.5" r="1.8" fill="#09090b"/><circle cx="16.6" cy="14.8" r=".6" fill="#fff"/><circle cx="24.6" cy="14.8" r=".6" fill="#fff"/><circle cx="16" cy="15.5" r="3.8" stroke="#F8717140" stroke-width=".7" fill="none"/><circle cx="24" cy="15.5" r="3.8" stroke="#A78BFA40" stroke-width=".7" fill="none"/><path d="M17 21Q20 24 23 21" stroke="#6D4C41" stroke-width=".9" fill="none" stroke-linecap="round"/><path d="M12 26Q14 24.5 17 24L20 25L23 24Q26 24.5 28 26L29 37L20 39L11 37Z" fill="#09090b"/><circle cx="17" cy="29" r="2" fill="#F87171"/><text x="17" y="30.2" text-anchor="middle" fill="#fff" font-size="2.5" font-weight="900">N</text></svg>
        </div>
        <div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="font-size:14px;font-weight:700;color:#fafafa">Nyx</span>
            <span id="agent-count" class="mono" style="font-size:10px;color:#34d399;font-weight:500"></span>
          </div>
          <div id="greeting" style="font-size:17px;color:#d4d4d8;line-height:1.6;font-weight:400"></div>
        </div>
      </div>
    </div>

    <!-- Search -->
    <div class="fi cd search-wrap" style="margin-bottom:32px">
      <input id="q" class="search-input" placeholder="Ask anything about tokens, protocols, security, research..." onkeydown="if(event.key==='Enter')go()">
      <button id="ask-btn" class="search-btn inactive" onclick="go()">Ask Nyx</button>
    </div>

    <!-- Prompts -->
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <button class="cd prompt-btn" style="flex:2" onclick="run('Is $XYBER safe? Full security audit')"><div class="prompt-title">Security audit</div><div class="prompt-desc">Contract vulnerabilities, rug signals, and code quality analysis</div></button>
      <button class="cd prompt-btn" style="flex:3" onclick="run('Find undervalued DeFi tokens with strong dev activity')"><div class="prompt-title">Alpha discovery</div><div class="prompt-desc">Market data cross-referenced with developer activity and research papers</div></button>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:40px">
      <button class="cd prompt-btn" style="flex:3" onclick="run('Latest breakthroughs in zero knowledge proof research')"><div class="prompt-title">Research briefing</div><div class="prompt-desc">Academic papers and live web coverage synthesized into key insights</div></button>
      <button class="cd prompt-btn" style="flex:2" onclick="run('Compare Solana vs Arbitrum for deploying DeFi')"><div class="prompt-title">Protocol comparison</div><div class="prompt-desc">Security, performance, and ecosystem analysis</div></button>
    </div>

    <!-- Activity feed -->
    <div id="history-section" style="display:none;margin-bottom:32px">
      <div class="sec-label">Recent activity</div>
      <div id="history-list"></div>
    </div>

    <!-- How it works -->
    <div style="border-top:1px solid #18181b;padding-top:28px;display:flex;gap:40px">
      <div class="how-step"><div class="how-num">01</div><div class="how-title">You ask a question</div><div class="how-desc">Anything about tokens, protocols, security, research, or code</div></div>
      <div class="how-step"><div class="how-num">02</div><div class="how-title">Nyx coordinates agents</div><div class="how-desc">Routes your query to every relevant agent on the network in parallel</div></div>
      <div class="how-step"><div class="how-num">03</div><div class="how-title">Verified answer</div><div class="how-desc">LLM synthesizes results. Every agent paid via x402. Receipts on Base</div></div>
    </div>
  </div>

  <!-- LOADING -->
  <div id="loading" style="display:none;padding-top:72px;padding-bottom:40px">
    <div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:36px">
      <div style="flex-shrink:0;animation:br 2s ease-in-out infinite">
        <svg width="36" height="36" viewBox="0 0 40 40" fill="none"><defs><linearGradient id="nl" x1="8" y1="4" x2="32" y2="20"><stop offset="0%" stop-color="#F87171"/><stop offset="50%" stop-color="#A78BFA"/><stop offset="100%" stop-color="#F472B6"/></linearGradient></defs><ellipse cx="20" cy="13" rx="13" ry="11" fill="url(#nl)"/><ellipse cx="20" cy="17" rx="10" ry="11" fill="#8D6E63"/><circle cx="16" cy="15.5" r="1.8" fill="#09090b"/><circle cx="24" cy="15.5" r="1.8" fill="#09090b"/><circle cx="16.6" cy="14.8" r=".6" fill="#fff"/><circle cx="24.6" cy="14.8" r=".6" fill="#fff"/><path d="M17 21Q20 24 23 21" stroke="#6D4C41" stroke-width=".9" fill="none" stroke-linecap="round"/><path d="M12 26Q14 24.5 17 24L20 25L23 24Q26 24.5 28 26L29 37L20 39L11 37Z" fill="#09090b"/><circle cx="17" cy="29" r="2" fill="#F87171"/><text x="17" y="30.2" text-anchor="middle" fill="#fff" font-size="2.5" font-weight="900">N</text></svg>
      </div>
      <div>
        <div style="font-size:15px;font-weight:700;color:#fafafa;margin-bottom:4px">Coordinating agents...</div>
        <div style="font-size:13px;color:#52525b">Querying the network and settling x402 payments on Base</div>
      </div>
    </div>
    <div id="agent-progress" style="padding-left:52px;margin-bottom:36px"></div>
    <div id="timer" class="mono" style="padding-left:52px;font-size:28px;font-weight:300;color:#18181b">0.0s</div>
  </div>

  <!-- RESULTS -->
  <div id="results" style="display:none;padding-top:20px;padding-bottom:48px">
    <div class="cd search-wrap" style="margin-bottom:24px;padding:3px 3px 3px 16px">
      <input id="q2" style="flex:1;border:none;background:transparent;font-size:13px;color:#fafafa;font-family:inherit;padding:10px 0" onkeydown="if(event.key==='Enter')go()">
      <button onclick="go()" style="background:#fafafa;color:#09090b;border:none;padding:9px 20px;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit">Ask</button>
    </div>
    <div id="answer-card"></div>
    <div id="meta-line" class="meta-line mono"></div>
    <div id="sources-section"></div>
    <div id="receipts-section"></div>
    <div style="padding-top:8px"><button onclick="reset()" class="cd" style="background:transparent;color:#52525b;padding:10px 22px;font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;border:1px solid #1c1c22;border-radius:12px">&larr; New query</button></div>
  </div>
</main>

<footer><div class="foot-inner">
  <span>Nexus &middot; Compound intelligence for Xyber</span>
  <span>x402 &middot; Base L2</span>
</div></footer>

<script>
const API='';
const AGENTS={quill:{n:'Quill',r:'Security',c:'#F87171'},tavily:{n:'Tavily',r:'Intel',c:'#60A5FA'},arxiv:{n:'ArXiv',r:'Research',c:'#FBBF24'},wikipedia:{n:'Wikipedia',r:'Knowledge',c:'#C084FC'},gitparser:{n:'GitParser',r:'Code',c:'#34D399'}};
const GREETINGS=["I have access to every agent on the Xyber network. Ask me anything.","Security, research, code, market data \u2014 I pull from all of them at once.","Every answer I give is backed by onchain payment receipts. No trust required.","I don't guess. I coordinate agents and synthesize what they find."];
let hist=[],tmr=null,t0=0;

document.getElementById('greeting').textContent=GREETINGS[Math.floor(Math.random()*GREETINGS.length)];

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}

// Ping health
async function ping(){
  try{const r=await fetch(API+'/api/health');const d=await r.json();
    const c=d.agents_healthy||d.agents_registered||0;
    document.getElementById('agent-count').textContent=c+' agents connected';
    document.getElementById('nav-dot').style.background='#34d399';
    document.getElementById('nav-status').style.color='#34d399';
    document.getElementById('nav-status').textContent='BASE';
  }catch(e){document.getElementById('nav-dot').style.background='#ef4444';document.getElementById('nav-status').style.color='#ef4444'}
}
ping();setInterval(ping,25000);

// Update nav stats
function updateNav(){
  const tp=hist.reduce((s,h)=>s+(h.d.x402_payments||0),0);
  if(hist.length)document.getElementById('nav-stats').textContent=hist.length+' queries \u00b7 $'+(tp*.01).toFixed(2)+' paid';
}

// Input handler
document.getElementById('q').addEventListener('input',function(){
  const b=document.getElementById('ask-btn');
  if(this.value.trim()){b.className='search-btn active'}else{b.className='search-btn inactive'}
});

function run(prompt){document.getElementById('q').value=prompt;go()}

async function go(){
  const el=document.getElementById('q');const g=el.value.trim();if(!g)return;
  document.getElementById('q2').value=g;
  show('loading');
  // Build agent progress
  let html='';
  Object.entries(AGENTS).forEach(([k,a],i)=>{
    html+=`<div class="agent-row" style="margin-bottom:10px;animation:fi .3s ease both;animation-delay:${i*.06}s">
      <div class="agent-dot" style="background:${a.c};animation:dt 1.4s ease infinite ${i*.2}s"></div>
      <span class="agent-name">${a.n}</span>
      <div class="shimmer" style="width:80px;background:linear-gradient(90deg,#18181b 25%,${a.c}30 50%,#18181b 75%);background-size:200% 100%"></div>
    </div>`;
  });
  document.getElementById('agent-progress').innerHTML=html;
  t0=Date.now();
  tmr=setInterval(()=>{document.getElementById('timer').textContent=((Date.now()-t0)/1000).toFixed(1)+'s'},50);
  try{
    const r=await fetch(API+'/hybrid/orchestrate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({goal:g})});
    const d=await r.json();clearInterval(tmr);
    if(d.error){show('home');alert('Error: '+d.error);return}
    hist.unshift({g,d,t:Date.now()});if(hist.length>20)hist.pop();
    updateNav();renderHistory();renderResults(d);show('results');ping();
  }catch(e){clearInterval(tmr);show('home');alert('Network error: '+e.message)}
}

function reset(){show('home');document.getElementById('q').value='';document.getElementById('q').focus()}

function show(v){
  document.getElementById('home').style.display=v==='home'?'block':'none';
  document.getElementById('loading').style.display=v==='loading'?'block':'none';
  document.getElementById('results').style.display=v==='results'?'block':'none';
}

function renderHistory(){
  if(!hist.length){document.getElementById('history-section').style.display='none';return}
  document.getElementById('history-section').style.display='block';
  let h='';
  hist.slice(0,4).forEach((item,i)=>{
    h+=`<button onclick="document.getElementById('q').value='${esc(item.g.replace(/'/g,"\\'"))}';go()" style="display:flex;align-items:center;gap:10px;width:100%;padding:10px 0;background:transparent;border:none;border-bottom:1px solid #18181b;cursor:pointer;font-family:inherit;text-align:left">
      <span style="font-size:12px;color:#71717a;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(item.g)}</span>
      <span class="mono" style="font-size:10px;color:#27272a">${item.d.agents_called} agents &middot; $${(item.d.x402_payments*.01).toFixed(2)}</span>
    </button>`;
  });
  document.getElementById('history-list').innerHTML=h;
}

function dtx(b){try{return JSON.parse(atob(b))}catch{return null}}
function pd(raw){
  if(!raw)return'';
  let t=typeof raw==='string'?raw:raw.raw_text||'';
  if(!t&&typeof raw==='object'){const a=raw.result||raw.results;if(Array.isArray(a))return a.slice(0,2).map(x=>typeof x==='string'?x:(x.title||x.name||'')).filter(Boolean).join(' \u00b7 ');return Object.entries(raw).slice(0,2).map(([k,v])=>k+': '+String(v).slice(0,50)).join(' \u00b7 ')}
  const m=t.match(/data:\s*(\{.+)/s);
  if(m){try{const o=JSON.parse(m[1]),inner=o?.result?.content?.[0]?.text;if(inner){try{const j=JSON.parse(inner),a=j.result||j.results;if(Array.isArray(a))return a.slice(0,2).map(x=>typeof x==='string'?x:(x.title||x.name||'')).filter(Boolean).join(' \u00b7 ');return Object.entries(j).slice(0,2).map(([k,v])=>k+': '+String(v).slice(0,50)).join(' \u00b7 ')}catch{return inner.slice(0,140)}}}catch{}}
  return t.slice(0,140);
}

function renderResults(d){
  // Answer
  let recs='';
  if(d.recommendations&&d.recommendations.length){
    recs='<div style="margin-top:18px;padding-left:14px;border-left:2px solid #27272a">';
    d.recommendations.forEach(r=>{recs+=`<div style="font-size:13px;color:#71717a;line-height:1.6;margin-bottom:3px">&rarr; ${esc(r)}</div>`});
    recs+='</div>';
  }
  document.getElementById('answer-card').innerHTML=`
    <div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:20px">
      <svg width="30" height="30" viewBox="0 0 40 40" fill="none"><defs><linearGradient id="nr" x1="8" y1="4" x2="32" y2="20"><stop offset="0%" stop-color="#F87171"/><stop offset="50%" stop-color="#A78BFA"/><stop offset="100%" stop-color="#F472B6"/></linearGradient></defs><ellipse cx="20" cy="13" rx="13" ry="11" fill="url(#nr)"/><ellipse cx="20" cy="17" rx="10" ry="11" fill="#8D6E63"/><circle cx="16" cy="15.5" r="1.8" fill="#09090b"/><circle cx="24" cy="15.5" r="1.8" fill="#09090b"/><circle cx="16.6" cy="14.8" r=".6" fill="#fff"/><circle cx="24.6" cy="14.8" r=".6" fill="#fff"/><path d="M17 21Q20 24 23 21" stroke="#6D4C41" stroke-width=".9" fill="none" stroke-linecap="round"/><path d="M12 26Q14 24.5 17 24L20 25L23 24Q26 24.5 28 26L29 37L20 39L11 37Z" fill="#09090b"/><circle cx="17" cy="29" r="2" fill="#F87171"/><text x="17" y="30.2" text-anchor="middle" fill="#fff" font-size="2.5" font-weight="900">N</text></svg>
      <div class="result-card" style="flex:1">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
          <span style="font-size:13px;font-weight:700;color:#fafafa">Nyx</span>
          <span style="font-size:9px;color:#71717a;background:#18181b;padding:3px 8px;border-radius:4px;font-weight:500">${esc((d.task_type||'analysis').replace(/_/g,' '))}</span>
          <span class="mono" style="font-size:9px;color:#3f3f46">${(d.confidence_score*100).toFixed(0)}% confidence &middot; ${d.agents_called} agents</span>
        </div>
        <div style="font-size:15px;color:#d4d4d8;line-height:1.78">${esc(d.summary||'')}</div>
        ${recs}
      </div>
    </div>`;

  // Meta
  document.getElementById('meta-line').innerHTML=
    `${d.agents_called} agents <span style="color:#27272a">&middot;</span> ${d.x402_payments} payments <span style="color:#27272a">&middot;</span> <span style="color:#34d399">$${(d.x402_payments*.01).toFixed(3)} USDC</span> <span style="color:#27272a">&middot;</span> ${(d.execution_time_ms/1000).toFixed(1)}s <span style="color:#27272a">&middot;</span> settled on Base`;

  // Sources
  if(d.details&&Object.keys(d.details).length){
    let s='<div class="sec-label">Sources</div>';
    Object.entries(d.details).forEach(([key,val])=>{
      const log=(d.execution_log||[]).find(l=>l.subtask_id===key);
      const ak=log?.agent||key;const a=AGENTS[ak]||{n:ak,c:'#52525b',r:''};
      const tx=log?.x402_tx?dtx(log.x402_tx):null;const pv=pd(val);
      let right='';
      if(log?.time_ms)right+=`<span class="mono" style="font-size:10px;color:#27272a">${(log.time_ms/1000).toFixed(1)}s</span>`;
      if(tx)right+=` <a href="https://basescan.org/tx/${tx.transaction}" target="_blank" style="font-size:9px;font-weight:600;color:#34d399;text-decoration:none;background:#34d39908;padding:2px 8px;border-radius:4px">Verified &nearr;</a>`;
      s+=`<div class="cd" style="padding:14px 18px;display:flex;align-items:flex-start;gap:12px;margin-bottom:6px">
        <div class="source-bar" style="height:22px;background:${a.c};margin-top:2px"></div>
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:${pv?'4':'0'}px">
            <span style="font-size:13px;font-weight:600;color:#d4d4d8">${a.n}</span>
            <span style="font-size:10px;color:#52525b">${a.r}</span>
            <div style="margin-left:auto;display:flex;gap:8px;align-items:center">${right}</div>
          </div>
          ${pv?`<div style="font-size:12px;color:#52525b;line-height:1.45;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(pv)}</div>`:''}
        </div>
      </div>`;
    });
    document.getElementById('sources-section').innerHTML=s;
  }else{document.getElementById('sources-section').innerHTML=''}

  // Receipts
  if(d.execution_log&&d.execution_log.some(l=>l.x402_tx)){
    let r='<div class="sec-label" style="margin-top:24px">Onchain Receipts</div><div style="display:flex;gap:6px;flex-wrap:wrap">';
    d.execution_log.filter(l=>l.x402_tx).forEach(log=>{
      const tx=dtx(log.x402_tx);const a=AGENTS[log.agent]||{n:log.agent,c:'#52525b'};
      if(tx)r+=`<a href="https://basescan.org/tx/${tx.transaction}" target="_blank" class="cd receipt-chip">
        <div style="width:3px;height:12px;border-radius:1px;background:${a.c};opacity:.5"></div>
        <span style="font-weight:600;color:#a1a1aa">${a.n}</span>
        <span class="mono" style="color:#27272a">${tx.transaction?.slice(0,10)}&hellip;</span>
        <span style="font-weight:600;color:#34d399">$0.01</span>
      </a>`;
    });
    r+='</div>';
    document.getElementById('receipts-section').innerHTML=r;
  }else{document.getElementById('receipts-section').innerHTML=''}
}
</script>
</body>
</html>
"""
write('src/mcp_server_nexus/dashboard.html', DASHBOARD_HTML)

# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# tests/__init__.py
# ──────────────────────────────────────────────
write("tests/__init__.py", "")

print("\\nNEXUS: All files generated successfully!")
print("Files created in src/mcp_server_nexus/")
