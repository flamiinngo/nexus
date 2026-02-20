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
    return ExecutionPlan(goal=goal, task_type=TaskType.TOKEN_EVALUATION, execution_mode=ExecutionMode.PARALLEL, subtasks=[
        Subtask(id="security", agent_name="quill", tool_name="search_token", description="Search for token info", parameters={"query": _extract_token_query(goal)}),
        Subtask(id="sentiment", agent_name="tavily", tool_name="web_search", description="Search for sentiment and news", parameters={"query": f"crypto token {goal} sentiment news"}),
        Subtask(id="code", agent_name="gitparser", tool_name="parse_github", description="Analyze codebase and developer activity", parameters={"query": goal}),
    ], synthesis_prompt="Synthesize security + sentiment + code into unified token evaluation with score 0-100 and BUY/HOLD/AVOID.")

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

    prompt = f"""You are NEXUS, an AI orchestration engine on the Xyber protocol. You just queried multiple AI agents to answer a user's goal. Synthesize their responses into ONE clear, informative answer.

USER GOAL: {goal}
TASK TYPE: {task_type}
AGENTS RESPONDED: {len(agent_data)}
CONFIDENCE: {confidence:.0%}

AGENT DATA:
{context}

Instructions:
- Write a clear 2-4 sentence summary that directly answers the user's goal
- If this is a token/security evaluation, include a risk assessment
- Be specific — cite data from the agents (prices, paper titles, findings)
- End with 1-2 actionable recommendations
- Do NOT mention that you are an AI or that agents were called — just deliver the answer

Respond in this exact JSON format:
{{"summary": "your 2-4 sentence synthesis here", "recommendations": ["recommendation 1", "recommendation 2"]}}"""

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
write("src/mcp_server_nexus/api_routers/__init__.py", '''"""API-only endpoints — REST, not MCP."""
from fastapi import APIRouter
from mcp_server_nexus.orchestrator.registry import AGENT_REGISTRY, get_all_healthy_agents

router = APIRouter(tags=["nexus-api"])

@router.get("/health", operation_id="health_check")
async def health():
    h = get_all_healthy_agents()
    return {"status": "operational", "service": "NEXUS", "version": "0.1.0", "agents_registered": len(AGENT_REGISTRY), "agents_healthy": len(h)}

@router.get("/stats", operation_id="get_stats")
async def stats():
    return {"total_calls": sum(a.call_count for a in AGENT_REGISTRY.values()), "agents": {n: {"calls": a.call_count, "avg_ms": round(a.avg_response_ms,1), "healthy": a.healthy} for n, a in AGENT_REGISTRY.items()}}

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
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS — Compound Intelligence Engine</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;background:#09090B;color:#A1A1AA;font-family:'Space Grotesk','Inter',sans-serif}
::placeholder{color:#3F3F46}
input:focus{outline:none}
@keyframes up{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes breathe{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes blink{0%,100%{opacity:.2}50%{opacity:1}}
@keyframes glow{0%,100%{box-shadow:0 0 20px #EF444420}50%{box-shadow:0 0 40px #EF444440}}
.card{background:linear-gradient(135deg,#18181B,#0F0F12);border:1px solid #27272A;border-radius:16px;transition:all .25s}
.card:hover{border-color:#EF444440;transform:translateY(-2px);box-shadow:0 8px 32px #EF444410}
.wrap{max-width:1100px;margin:0 auto;padding:0 32px}
nav{position:sticky;top:0;z-index:50;background:#09090Bf0;backdrop-filter:blur(20px);border-bottom:1px solid #18181B}
.nav-inner{height:56px;display:flex;align-items:center;justify-content:space-between}
.btn{padding:12px 28px;border-radius:10px;background:#EF4444;color:#fff;border:none;font-size:13px;font-weight:700;cursor:pointer;font-family:inherit;letter-spacing:.5px;transition:all .2s}
.btn:disabled{background:#27272A;color:#52525B;cursor:default}
.input-wrap{background:#18181B;border-radius:14px;border:1px solid #27272A;display:flex;align-items:center;padding:5px 5px 5px 20px}
.input-wrap input{flex:1;border:none;background:transparent;font-size:14px;color:#FAFAFA;font-family:inherit;padding:12px 0}
.hidden{display:none}
footer{border-top:1px solid #18181B;margin-top:32px}
</style>
</head>
<body>

<!-- NAV -->
<nav>
<div class="wrap nav-inner">
  <div style="display:flex;align-items:center;gap:12px">
    <svg width="30" height="30" viewBox="0 0 40 40" fill="none"><defs><linearGradient id="hm2" x1="8" y1="4" x2="32" y2="20"><stop offset="0%" stop-color="#EF4444"/><stop offset="50%" stop-color="#8B5CF6"/><stop offset="100%" stop-color="#EC4899"/></linearGradient></defs><ellipse cx="20" cy="13" rx="13" ry="11" fill="url(#hm2)"/><ellipse cx="20" cy="17" rx="10" ry="11" fill="#8D6E63"/><circle cx="16" cy="15.5" r="1.8" fill="#18181B"/><circle cx="24" cy="15.5" r="1.8" fill="#18181B"/><circle cx="16.6" cy="14.8" r=".6" fill="white"/><circle cx="24.6" cy="14.8" r=".6" fill="white"/><circle cx="16" cy="15.5" r="3.8" stroke="#EF444460" stroke-width=".7" fill="none"/><circle cx="24" cy="15.5" r="3.8" stroke="#A855F760" stroke-width=".7" fill="none"/><path d="M17 21Q20 24 23 21" stroke="#6D4C41" stroke-width=".9" fill="none" stroke-linecap="round"/><path d="M12 26Q14 24.5 17 24L20 25L23 24Q26 24.5 28 26L29 37L20 39L11 37Z" fill="#09090B"/><circle cx="17" cy="29" r="2" fill="#EF4444"/><text x="17" y="30.2" text-anchor="middle" fill="white" font-size="2.5" font-weight="900">N</text></svg>
    <span style="font-size:16px;font-weight:700;color:#FAFAFA;letter-spacing:2px">NEXUS</span>
    <span style="font-size:10px;color:#3F3F46;font-weight:500;margin-left:4px">XYBER PROTOCOL</span>
  </div>
  <div style="display:flex;align-items:center;gap:6px">
    <div style="width:6px;height:6px;border-radius:50%;background:#EF4444;animation:glow 2s ease infinite"></div>
    <span style="font-size:10px;font-weight:700;color:#EF4444;letter-spacing:1px">LIVE ON BASE</span>
  </div>
</div>
</nav>

<main class="wrap">

<!-- HERO -->
<div id="hero" style="padding-top:56px;padding-bottom:48px;animation:up .5s ease">
  <div style="display:flex;align-items:center;gap:40px">
    <div style="flex:1">
      <div style="font-size:11px;font-weight:700;color:#EF4444;letter-spacing:4px;margin-bottom:20px">COMPOUND INTELLIGENCE ENGINE</div>
      <h1 style="font-size:52px;font-weight:900;color:#FAFAFA;line-height:1.05;margin:0 0 24px;font-family:'Inter',sans-serif;letter-spacing:-1px">DON'T TRUST<br>ONE AGENT.<br><span style="color:#EF4444">TRUST ALL OF THEM.</span></h1>
      <p style="font-size:16px;color:#71717A;line-height:1.7;margin:0 0 36px;max-width:420px">Nyx queries every AI agent in the Xyber ecosystem simultaneously, pays each one via x402 micropayments, and synthesizes intelligence no single agent can match.</p>
      <div class="input-wrap" style="max-width:480px">
        <input id="goalInput" placeholder="What do you want to know?">
        <button class="btn" id="askBtn" onclick="doQuery()">ASK NYX</button>
      </div>
    </div>
    <div style="flex-shrink:0;animation:breathe 4s ease-in-out infinite">
      <svg width="220" height="220" viewBox="0 0 200 200" fill="none"><defs><linearGradient id="h1" x1="58" y1="28" x2="142" y2="88"><stop offset="0%" stop-color="#EF4444"/><stop offset="50%" stop-color="#8B5CF6"/><stop offset="100%" stop-color="#EC4899"/></linearGradient><linearGradient id="c1" x1="60" y1="95" x2="140" y2="180"><stop offset="0%" stop-color="#18181B"/><stop offset="100%" stop-color="#09090B"/></linearGradient><radialGradient id="rg" cx="50%" cy="30%" r="45%"><stop offset="0%" stop-color="#EF444418"/><stop offset="100%" stop-color="#EF444400"/></radialGradient><filter id="gx"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="100" cy="85" r="75" fill="url(#rg)"/><g opacity=".2" filter="url(#gx)"><line x1="36" y1="44" x2="64" y2="62" stroke="#EF4444" stroke-width="1"/><circle cx="35" cy="46" r="3.5" fill="#EF4444"/><line x1="164" y1="44" x2="136" y2="62" stroke="#A855F7" stroke-width="1"/><circle cx="165" cy="46" r="3.5" fill="#A855F7"/><line x1="24" y1="82" x2="58" y2="78" stroke="#10B981" stroke-width="1"/><circle cx="25" cy="82" r="2.5" fill="#10B981"/><line x1="176" y1="82" x2="142" y2="78" stroke="#F59E0B" stroke-width="1"/><circle cx="175" cy="82" r="2.5" fill="#F59E0B"/><line x1="40" y1="116" x2="68" y2="96" stroke="#3B82F6" stroke-width="1"/><circle cx="40" cy="114" r="2.5" fill="#3B82F6"/><line x1="160" y1="116" x2="132" y2="96" stroke="#EF4444" stroke-width="1"/><circle cx="160" cy="114" r="2.5" fill="#EF4444"/></g><ellipse cx="100" cy="58" rx="43" ry="38" fill="url(#h1)"/><path d="M57 58Q53 86 62 102Q57 78 59 63Z" fill="url(#h1)" opacity=".7"/><path d="M143 58Q147 86 138 102Q143 78 141 63Z" fill="url(#h1)" opacity=".7"/><ellipse cx="100" cy="72" rx="33" ry="36" fill="#8D6E63"/><ellipse cx="86" cy="67" rx="6.5" ry="5" fill="white"/><ellipse cx="114" cy="67" rx="6.5" ry="5" fill="white"/><circle cx="87" cy="67" r="3.2" fill="#18181B"/><circle cx="115" cy="67" r="3.2" fill="#18181B"/><circle cx="88.2" cy="66" r="1.3" fill="white"/><circle cx="116.2" cy="66" r="1.3" fill="white"/><path d="M79 61Q87 57 94 60" stroke="#5D4037" stroke-width="1.8" fill="none" stroke-linecap="round"/><path d="M106 60Q113 57 121 61" stroke="#5D4037" stroke-width="1.8" fill="none" stroke-linecap="round"/><path d="M97 74Q100 79 103 74" stroke="#795548" stroke-width="1" fill="none"/><path d="M90 83Q100 88 110 83" stroke="#6D4C41" stroke-width="1.8" fill="none" stroke-linecap="round"/><circle cx="86" cy="67" r="11" stroke="#EF4444" stroke-width="1.5" fill="none" opacity=".5"/><circle cx="114" cy="67" r="11" stroke="#A855F7" stroke-width="1.5" fill="none" opacity=".5"/><line x1="97" y1="67" x2="103" y2="67" stroke="#71717A" stroke-width="1"/><line x1="75" y1="66" x2="66" y2="62" stroke="#71717A" stroke-width="1"/><line x1="125" y1="66" x2="134" y2="62" stroke="#71717A" stroke-width="1"/><path d="M64 100Q67 94 80 93L100 95L120 93Q133 94 136 100L142 150Q142 160 132 163L100 167L68 163Q58 160 58 150Z" fill="url(#c1)"/><path d="M81 93L95 108L100 95L105 108L119 93" stroke="#27272A" stroke-width="1.2" fill="none"/><g opacity=".4" filter="url(#gx)"><line x1="71" y1="108" x2="71" y2="145" stroke="#EF4444" stroke-width=".9"/><line x1="71" y1="126" x2="82" y2="126" stroke="#EF4444" stroke-width=".9"/><circle cx="82" cy="126" r="2" fill="#EF4444"/><line x1="129" y1="108" x2="129" y2="145" stroke="#EF4444" stroke-width=".9"/><line x1="129" y1="118" x2="118" y2="118" stroke="#EF4444" stroke-width=".9"/><circle cx="118" cy="118" r="2" fill="#EF4444"/></g><circle cx="87" cy="112" r="7" fill="#EF4444"/><text x="87" y="115.5" text-anchor="middle" fill="white" font-size="9" font-weight="900">N</text><g transform="translate(131,114)rotate(8)"><rect width="24" height="32" rx="3" fill="#09090B" stroke="#EF444435" stroke-width="1"/><rect x="3" y="4" width="18" height="8" rx="1.5" fill="#EF444410"/></g><ellipse cx="142" cy="130" rx="9" ry="5.5" fill="#8D6E63" transform="rotate(8,142,130)"/></svg>
    </div>
  </div>

  <!-- PROMPTS -->
  <div style="margin-top:48px">
    <div style="font-size:11px;font-weight:700;color:#3F3F46;letter-spacing:3px;margin-bottom:16px">TRY SOMETHING REAL</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
      <button class="card" onclick="doQuery('Is $XYBER safe? Run a full security check')" style="padding:20px;cursor:pointer;text-align:left;font-family:inherit">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px"><span style="font-size:16px">🛡</span><span style="font-size:10px;font-weight:700;color:#EF4444;letter-spacing:2px">SECURITY</span></div>
        <div style="font-size:13px;color:#A1A1AA;line-height:1.5">Is $XYBER safe? Run a full security check</div>
      </button>
      <button class="card" onclick="doQuery('Find undervalued DeFi tokens below $500M mcap')" style="padding:20px;cursor:pointer;text-align:left;font-family:inherit">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px"><span style="font-size:16px">📡</span><span style="font-size:10px;font-weight:700;color:#EF4444;letter-spacing:2px">ALPHA</span></div>
        <div style="font-size:13px;color:#A1A1AA;line-height:1.5">Find undervalued DeFi tokens below $500M mcap</div>
      </button>
      <button class="card" onclick="doQuery('Latest breakthroughs in zero knowledge proofs')" style="padding:20px;cursor:pointer;text-align:left;font-family:inherit">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px"><span style="font-size:16px">🔬</span><span style="font-size:10px;font-weight:700;color:#EF4444;letter-spacing:2px">RESEARCH</span></div>
        <div style="font-size:13px;color:#A1A1AA;line-height:1.5">Latest breakthroughs in zero knowledge proofs</div>
      </button>
    </div>
  </div>

  <!-- STATS -->
  <div style="display:flex;gap:0;margin-top:40px;border-top:1px solid #18181B;border-bottom:1px solid #18181B">
    <div style="flex:1;padding:20px 0;text-align:center;border-right:1px solid #18181B"><div style="font-size:22px;font-weight:900;color:#FAFAFA;font-family:'Inter',sans-serif">5+</div><div style="font-size:9px;font-weight:700;color:#3F3F46;letter-spacing:2px;margin-top:4px">AI AGENTS</div></div>
    <div style="flex:1;padding:20px 0;text-align:center;border-right:1px solid #18181B"><div style="font-size:22px;font-weight:900;color:#FAFAFA;font-family:'Inter',sans-serif">$0.03</div><div style="font-size:9px;font-weight:700;color:#3F3F46;letter-spacing:2px;margin-top:4px">PER QUERY</div></div>
    <div style="flex:1;padding:20px 0;text-align:center;border-right:1px solid #18181B"><div style="font-size:22px;font-weight:900;color:#FAFAFA;font-family:'Inter',sans-serif">x402</div><div style="font-size:9px;font-weight:700;color:#3F3F46;letter-spacing:2px;margin-top:4px">PROTOCOL</div></div>
    <div style="flex:1;padding:20px 0;text-align:center;border-right:1px solid #18181B"><div style="font-size:22px;font-weight:900;color:#FAFAFA;font-family:'Inter',sans-serif">BASE</div><div style="font-size:9px;font-weight:700;color:#3F3F46;letter-spacing:2px;margin-top:4px">SETTLEMENT</div></div>
    <div style="flex:1;padding:20px 0;text-align:center"><div style="font-size:22px;font-weight:900;color:#FAFAFA;font-family:'Inter',sans-serif">∞</div><div style="font-size:9px;font-weight:700;color:#3F3F46;letter-spacing:2px;margin-top:4px">SCALABLE</div></div>
  </div>
</div>

<!-- LOADING -->
<div id="loading" class="hidden" style="text-align:center;padding:72px 0;animation:up .3s ease">
  <div style="animation:breathe 2s ease-in-out infinite;margin-bottom:32px">
    <svg width="140" height="140" viewBox="0 0 200 200" fill="none"><defs><linearGradient id="h1b" x1="58" y1="28" x2="142" y2="88"><stop offset="0%" stop-color="#EF4444"/><stop offset="50%" stop-color="#8B5CF6"/><stop offset="100%" stop-color="#EC4899"/></linearGradient><linearGradient id="c1b" x1="60" y1="95" x2="140" y2="180"><stop offset="0%" stop-color="#18181B"/><stop offset="100%" stop-color="#09090B"/></linearGradient></defs><ellipse cx="100" cy="58" rx="43" ry="38" fill="url(#h1b)"/><path d="M57 58Q53 86 62 102Q57 78 59 63Z" fill="url(#h1b)" opacity=".7"/><path d="M143 58Q147 86 138 102Q143 78 141 63Z" fill="url(#h1b)" opacity=".7"/><ellipse cx="100" cy="72" rx="33" ry="36" fill="#8D6E63"/><ellipse cx="86" cy="67" rx="6.5" ry="5" fill="white"/><ellipse cx="114" cy="67" rx="6.5" ry="5" fill="white"/><circle cx="87" cy="67" r="3.2" fill="#18181B"/><circle cx="115" cy="67" r="3.2" fill="#18181B"/><circle cx="88.2" cy="66" r="1.3" fill="white"/><circle cx="116.2" cy="66" r="1.3" fill="white"/><path d="M79 61Q87 57 94 60" stroke="#5D4037" stroke-width="1.8" fill="none"/><path d="M106 60Q113 57 121 61" stroke="#5D4037" stroke-width="1.8" fill="none"/><path d="M89 83Q100 92 111 83" stroke="#6D4C41" stroke-width="1.8" fill="none" stroke-linecap="round"/><circle cx="86" cy="67" r="11" stroke="#EF4444" stroke-width="1.5" fill="none" opacity=".5"/><circle cx="114" cy="67" r="11" stroke="#A855F7" stroke-width="1.5" fill="none" opacity=".5"/><line x1="97" y1="67" x2="103" y2="67" stroke="#71717A" stroke-width="1"/><path d="M64 100Q67 94 80 93L100 95L120 93Q133 94 136 100L142 150Q142 160 132 163L100 167L68 163Q58 160 58 150Z" fill="url(#c1b)"/><circle cx="87" cy="112" r="7" fill="#EF4444"/><text x="87" y="115.5" text-anchor="middle" fill="white" font-size="9" font-weight="900">N</text></svg>
  </div>
  <div style="font-size:28px;font-weight:900;color:#FAFAFA;margin-bottom:8px;font-family:'Inter',sans-serif">NYX IS WORKING</div>
  <div style="font-size:13px;color:#52525B;margin-bottom:32px">Querying agents. Settling x402 payments on Base.</div>
  <div id="agentList" style="display:inline-flex;flex-direction:column;gap:8px;text-align:left"></div>
  <div id="timer" style="font-size:40px;font-weight:300;color:#27272A;font-family:monospace;margin-top:32px">0.0s</div>
</div>

<!-- RESULTS -->
<div id="results" class="hidden" style="padding:32px 0 64px;animation:up .4s ease">
  <div class="input-wrap" style="margin-bottom:32px">
    <input id="goalInput2" placeholder="Ask another question...">
    <button class="btn" onclick="doQuery()">ASK NYX</button>
  </div>
  <div id="synthesis" class="card" style="padding:28px;margin-bottom:20px"></div>
  <div id="statsRow" style="display:flex;gap:0;margin-bottom:24px;border:1px solid #27272A;border-radius:12px;overflow:hidden"></div>
  <div style="font-size:11px;font-weight:700;color:#3F3F46;letter-spacing:3px;margin-bottom:14px">AGENT SOURCES</div>
  <div id="sources" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:28px"></div>
  <div id="receiptsSection" class="hidden">
    <div style="font-size:11px;font-weight:700;color:#3F3F46;letter-spacing:3px;margin-bottom:14px">ONCHAIN RECEIPTS</div>
    <div id="receipts" style="display:flex;gap:10px;flex-wrap:wrap"></div>
  </div>
</div>

<!-- ERROR -->
<div id="errorBox" class="hidden" style="padding:24px;background:#18181B;border:1px solid #EF444430;border-radius:14px;margin:48px 0;text-align:center">
  <div id="errorMsg" style="font-size:13px;color:#EF4444"></div>
</div>

</main>

<footer>
<div class="wrap" style="padding:18px 32px;display:flex;justify-content:space-between;font-size:10px;color:#27272A">
  <span>NEXUS — THE COMPOUND INTELLIGENCE ENGINE</span>
  <div style="display:flex;gap:20px"><span>XYBER PROTOCOL</span><span>x402</span><span>BASE L2</span></div>
</div>
</footer>

<script>
const AGENTS = {
  quill:{l:"QUILL",r:"Security Audit",c:"#EF4444"},
  tavily:{l:"TAVILY",r:"Live Web Intel",c:"#3B82F6"},
  arxiv:{l:"ARXIV",r:"Research Papers",c:"#F59E0B"},
  wikipedia:{l:"WIKIPEDIA",r:"Knowledge",c:"#A855F7"},
  gitparser:{l:"GITPARSER",r:"Code Analysis",c:"#10B981"}
};

let timerInterval;

document.getElementById('goalInput').addEventListener('keydown',e=>{if(e.key==='Enter')doQuery()});
document.getElementById('goalInput2').addEventListener('keydown',e=>{if(e.key==='Enter')doQuery()});

async function doQuery(preset){
  const input1=document.getElementById('goalInput');
  const input2=document.getElementById('goalInput2');
  const goal=preset||input1.value||input2.value;
  if(!goal.trim())return;
  input1.value=goal;input2.value=goal;

  // Show loading
  document.getElementById('hero').classList.add('hidden');
  document.getElementById('results').classList.add('hidden');
  document.getElementById('errorBox').classList.add('hidden');
  document.getElementById('loading').classList.remove('hidden');

  // Agent list
  const al=document.getElementById('agentList');
  al.innerHTML=Object.entries(AGENTS).map(([k,a],i)=>`
    <div style="display:flex;align-items:center;gap:12px;animation:up .3s ease ${i*.08}s both">
      <div style="width:6px;height:6px;border-radius:50%;background:${a.c};animation:blink 1.2s ease infinite ${i*.15}s"></div>
      <span style="font-size:12px;font-weight:700;color:#FAFAFA;width:90px;letter-spacing:1px">${a.l}</span>
      <span style="font-size:11px;color:#3F3F46">${a.r}</span>
    </div>`).join('');

  // Timer
  const t0=Date.now();
  const timerEl=document.getElementById('timer');
  timerInterval=setInterval(()=>{timerEl.textContent=((Date.now()-t0)/1000).toFixed(1)+'s'},60);

  try{
    const r=await fetch('/hybrid/orchestrate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({goal})});
    const data=await r.json();
    clearInterval(timerInterval);
    if(data.error){showError(data.error);return}
    showResults(data);
  }catch(e){
    clearInterval(timerInterval);
    showError(e.message);
  }
}

function showError(msg){
  document.getElementById('loading').classList.add('hidden');
  document.getElementById('errorBox').classList.remove('hidden');
  document.getElementById('errorMsg').textContent=msg;
}

function showResults(d){
  document.getElementById('loading').classList.add('hidden');
  document.getElementById('results').classList.remove('hidden');

  const cost=(d.x402_payments*.01).toFixed(3);
  const conf=d.confidence_score!=null?(d.confidence_score*100).toFixed(0)+'%':'—';
  const taskType=(d.task_type||'').toUpperCase();

  // Synthesis
  let recHtml='';
  if(d.recommendations&&d.recommendations.length){
    recHtml='<div style="margin-top:16px;border-left:3px solid #EF4444;padding-left:16px">'+d.recommendations.map(r=>`<div style="font-size:13px;color:#EF4444;line-height:1.6;margin-bottom:4px">${esc(r)}</div>`).join('')+'</div>';
  }
  document.getElementById('synthesis').innerHTML=`
    <div style="display:flex;gap:20px;align-items:flex-start">
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none"><defs><linearGradient id="hm3" x1="8" y1="4" x2="32" y2="20"><stop offset="0%" stop-color="#EF4444"/><stop offset="50%" stop-color="#8B5CF6"/><stop offset="100%" stop-color="#EC4899"/></linearGradient></defs><ellipse cx="20" cy="13" rx="13" ry="11" fill="url(#hm3)"/><ellipse cx="20" cy="17" rx="10" ry="11" fill="#8D6E63"/><circle cx="16" cy="15.5" r="1.8" fill="#18181B"/><circle cx="24" cy="15.5" r="1.8" fill="#18181B"/><circle cx="16.6" cy="14.8" r=".6" fill="white"/><circle cx="24.6" cy="14.8" r=".6" fill="white"/><circle cx="16" cy="15.5" r="3.8" stroke="#EF444460" stroke-width=".7" fill="none"/><circle cx="24" cy="15.5" r="3.8" stroke="#A855F760" stroke-width=".7" fill="none"/><path d="M17 21Q20 24 23 21" stroke="#6D4C41" stroke-width=".9" fill="none" stroke-linecap="round"/><path d="M12 26Q14 24.5 17 24L20 25L23 24Q26 24.5 28 26L29 37L20 39L11 37Z" fill="#09090B"/><circle cx="17" cy="29" r="2" fill="#EF4444"/><text x="17" y="30.2" text-anchor="middle" fill="white" font-size="2.5" font-weight="900">N</text></svg>
      <div style="flex:1">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
          <span style="font-size:12px;font-weight:800;color:#FAFAFA;letter-spacing:1px">NYX</span>
          <span style="font-size:9px;font-weight:700;color:#EF4444;background:#EF444415;padding:3px 8px;border-radius:4px;letter-spacing:1px">${taskType}</span>
          <span style="font-size:9px;color:#3F3F46">${conf} CONFIDENCE</span>
        </div>
        <div style="font-size:15px;color:#D4D4D8;line-height:1.75">${esc(d.summary)}</div>
        ${recHtml}
      </div>
    </div>`;
  document.getElementById('synthesis').style.backgroundImage='radial-gradient(ellipse at top left,#EF444408 0%,transparent 40%)';

  // Stats
  document.getElementById('statsRow').innerHTML=[
    {l:'AGENTS',v:d.agents_called},{l:'PAYMENTS',v:d.x402_payments},{l:'COST',v:'$'+cost},{l:'SPEED',v:(d.execution_time_ms/1000).toFixed(1)+'s'}
  ].map((s,i)=>`<div style="flex:1;padding:16px 0;text-align:center;background:#18181B;${i<3?'border-right:1px solid #27272A':''}"><div style="font-size:9px;font-weight:700;color:#3F3F46;letter-spacing:2px;margin-bottom:4px">${s.l}</div><div style="font-size:20px;font-weight:900;color:#FAFAFA">${s.v}</div></div>`).join('');

  // Sources
  const src=document.getElementById('sources');
  src.innerHTML='';
  if(d.details){
    Object.entries(d.details).forEach(([key,val])=>{
      const log=d.execution_log?.find(l=>l.subtask_id===key);
      const a=AGENTS[log?.agent]||{l:log?.agent||key,r:'',c:'#71717A'};
      const tx=log?.x402_tx?dtx(log.x402_tx):null;
      const p=parse(val);
      let items=[];
      if(p&&p.t==='j'){const arr=p.d?.result||p.d?.results;if(Array.isArray(arr))items=arr.slice(0,2).map((x,i)=>({t:typeof x==='string'?x:(x.title||x.name||'#'+(i+1)),d:typeof x==='string'?'':(x.content||x.summary||x.url||'').slice(0,130)}));else if(typeof p.d==='object')items=Object.entries(p.d).slice(0,2).map(([k,v])=>({t:k,d:String(v).slice(0,100)}));}
      else if(p&&p.t==='t')items=[{t:'Response',d:String(p.d).slice(0,200)}];

      src.innerHTML+=`<div class="card" style="padding:14px 16px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
          <div style="display:flex;align-items:center;gap:8px"><div style="width:3px;height:20px;border-radius:2px;background:${a.c}"></div><span style="font-size:11px;font-weight:800;color:#FAFAFA;letter-spacing:1px">${a.l}</span><span style="font-size:9px;color:#3F3F46">${a.r}</span></div>
          <div style="display:flex;align-items:center;gap:6px">${log?.time_ms?`<span style="font-size:9px;color:#3F3F46;font-family:monospace">${(log.time_ms/1000).toFixed(1)}s</span>`:''}${tx?`<a href="https://basescan.org/tx/${tx.transaction}" target="_blank" style="font-size:8px;font-weight:800;color:#EF4444;background:#EF444415;padding:2px 8px;border-radius:4px;text-decoration:none;letter-spacing:1px">PAID ↗</a>`:''}</div>
        </div>
        ${items.map(it=>`<div style="margin-bottom:6px"><div style="font-size:11px;font-weight:600;color:#D4D4D8;line-height:1.4">${esc(it.t)}</div>${it.d?`<div style="font-size:10px;color:#52525B;margin-top:2px;line-height:1.5">${esc(it.d)}</div>`:''}</div>`).join('')}
      </div>`;
    });
  }

  // Receipts
  const hasReceipts=d.execution_log?.some(l=>l.x402_tx);
  if(hasReceipts){
    document.getElementById('receiptsSection').classList.remove('hidden');
    document.getElementById('receipts').innerHTML=d.execution_log.filter(l=>l.x402_tx).map(log=>{
      const tx=dtx(log.x402_tx);const a=AGENTS[log.agent]||{l:log.agent,c:'#71717A'};
      return tx?`<a href="https://basescan.org/tx/${tx.transaction}" target="_blank" class="card" style="display:flex;align-items:center;gap:10px;padding:10px 16px;text-decoration:none"><div style="width:3px;height:16px;border-radius:2px;background:${a.c}"></div><span style="font-size:10px;font-weight:700;color:#FAFAFA;letter-spacing:1px">${a.l}</span><span style="font-size:9px;color:#3F3F46;font-family:monospace">${tx.transaction?.slice(0,10)}…</span><span style="font-size:9px;font-weight:800;color:#EF4444">$0.01</span></a>`:'';
    }).join('');
  }
}

function parse(raw){
  if(!raw)return null;
  let t=typeof raw==='string'?raw:raw.raw_text||'';
  if(!t&&typeof raw==='object')return{t:'j',d:raw};
  const m=t.match(/data:\s*(\{.+)/s);
  if(m){try{const o=JSON.parse(m[1]),i=o?.result?.content?.[0]?.text;if(i){try{return{t:'j',d:JSON.parse(i)}}catch(e){return{t:'t',d:i}}}return{t:'j',d:o}}catch(e){}}
  return{t:'t',d:t.slice(0,400)};
}
function dtx(b){try{return JSON.parse(atob(b))}catch(e){return null}}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
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
