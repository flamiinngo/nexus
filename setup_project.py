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
    healthy: bool = True
    avg_response_ms: float = 0.0
    call_count: int = 0

AGENT_REGISTRY: dict[str, AgentProfile] = {
    "quill": AgentProfile(
        name="Quill",
        description="Token security analysis - audits smart contracts, detects rugs, rates safety",
        base_url="https://mcp-quill.xyber.inc",
        mcp_endpoint="/mcp",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.TOKEN_SECURITY, AgentCapability.TOKEN_ANALYSIS],
        tools={"analyze_token_security": "analyze_token", "check_contract": "check_contract"},
    ),
    "tavily": AgentProfile(
        name="Tavily",
        description="Web search and sentiment - finds news, social sentiment, market buzz",
        base_url="https://mcp-tavily.xyber.inc",
        mcp_endpoint="/mcp",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.WEB_SEARCH, AgentCapability.SENTIMENT_ANALYSIS, AgentCapability.NEWS_SEARCH],
        tools={"web_search": "search", "get_sentiment": "search"},
    ),
    "arxiv": AgentProfile(
        name="ArXiv",
        description="Research papers - finds academic research, whitepapers, technical analysis",
        base_url="https://mcp-arxiv.xyber.inc",
        mcp_endpoint="/mcp",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.RESEARCH_PAPERS],
        tools={"search_papers": "search_papers", "get_paper": "get_paper"},
    ),
    "wikipedia": AgentProfile(
        name="Wikipedia",
        description="Knowledge base - background info, definitions, context",
        base_url="https://mcp-wikipedia.xyber.inc",
        mcp_endpoint="/mcp",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.KNOWLEDGE_BASE],
        tools={"lookup": "search", "get_article": "get_article"},
    ),
    "gitparser": AgentProfile(
        name="GitParser",
        description="Code and repo analysis - evaluates codebases, commit history, developer activity",
        base_url="https://mcp-gitparser.xyber.inc",
        mcp_endpoint="/mcp",
        pricing_endpoint="/hybrid/pricing",
        capabilities=[AgentCapability.CODE_ANALYSIS, AgentCapability.REPO_ANALYSIS],
        tools={"analyze_repo": "analyze_repo", "get_repo_stats": "get_repo_stats"},
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

def _plan_token_eval(goal):
    return ExecutionPlan(goal=goal, task_type=TaskType.TOKEN_EVALUATION, execution_mode=ExecutionMode.PARALLEL, subtasks=[
        Subtask(id="security", agent_name="quill", tool_name="analyze_token_security", description="Analyze token contract security", parameters={"query": goal}),
        Subtask(id="sentiment", agent_name="tavily", tool_name="web_search", description="Search for sentiment and news", parameters={"query": f"crypto token {goal} sentiment news"}),
        Subtask(id="code", agent_name="gitparser", tool_name="analyze_repo", description="Analyze codebase and developer activity", parameters={"query": goal}),
    ], synthesis_prompt="Synthesize security + sentiment + code into unified token evaluation with score 0-100 and BUY/HOLD/AVOID.")

def _plan_opportunity(goal):
    return ExecutionPlan(goal=goal, task_type=TaskType.OPPORTUNITY_DISCOVERY, execution_mode=ExecutionMode.MIXED, subtasks=[
        Subtask(id="search", agent_name="tavily", tool_name="web_search", description="Search for trending opportunities", parameters={"query": f"crypto {goal} new opportunity 2025"}),
        Subtask(id="research", agent_name="arxiv", tool_name="search_papers", description="Find research on relevant protocols", parameters={"query": goal}),
        Subtask(id="security_check", agent_name="quill", tool_name="analyze_token_security", description="Security check discoveries", parameters={"query": goal}, depends_on=["search"]),
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
import asyncio, logging, time
from dataclasses import dataclass, field
from typing import Any
import httpx
from mcp_server_nexus.orchestrator.registry import AGENT_REGISTRY
from mcp_server_nexus.orchestrator.router import ExecutionMode, ExecutionPlan, Subtask

logger = logging.getLogger(__name__)

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
        logger.info("Plan done: %d tasks, %d ok, %.0fms", len(result.subtask_results), sum(1 for r in result.subtask_results if r.success), result.total_execution_time_ms)
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

    async def _run(self, subtask: Subtask) -> SubtaskResult:
        start = time.time()
        agent = AGENT_REGISTRY.get(subtask.agent_name)
        if not agent:
            return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=False, error=f"Agent not found")
        try:
            client = await self._get_client()
            mcp_url = f"{agent.base_url}{agent.mcp_endpoint}"
            remote_tool = agent.tools.get(subtask.tool_name, subtask.tool_name)
            body = {"jsonrpc": "2.0", "id": f"nexus-{subtask.id}-{int(time.time())}", "method": "tools/call", "params": {"name": remote_tool, "arguments": subtask.parameters}}
            headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
            logger.info("Calling %s -> %s", subtask.agent_name, remote_tool)
            resp = await client.post(mcp_url, json=body, headers=headers)
            ms = (time.time() - start) * 1000
            agent.call_count += 1
            agent.avg_response_ms = ((agent.avg_response_ms * (agent.call_count - 1)) + ms) / agent.call_count
            if resp.status_code == 402:
                return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=False, data={"payment_required": resp.json()}, error="x402 payment required", execution_time_ms=ms)
            if resp.status_code == 200:
                ptx = resp.headers.get("PAYMENT-RESPONSE")
                return SubtaskResult(subtask_id=subtask.id, agent_name=subtask.agent_name, success=True, data=resp.json(), execution_time_ms=ms, x402_payment_tx=ptx)
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
write("src/mcp_server_nexus/orchestrator/synthesizer.py", '''"""NEXUS Synthesizer — combines multi-agent results into unified intelligence."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
from mcp_server_nexus.orchestrator.executor import ExecutionResult, SubtaskResult
from mcp_server_nexus.orchestrator.router import TaskType

logger = logging.getLogger(__name__)

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

class Synthesizer:
    def synthesize(self, result: ExecutionResult) -> SynthesizedResult:
        log = self._build_log(result)
        contribs = self._build_contribs(result)
        plan = result.plan
        by_id = {r.subtask_id: r for r in result.subtask_results}
        ok = sum(1 for r in result.subtask_results if r.success)
        conf = ok / max(len(result.subtask_results), 1)

        if plan.task_type == TaskType.TOKEN_EVALUATION:
            sec = self._score(by_id.get("security"), 50)
            sent = self._score(by_id.get("sentiment"), 50)
            code = self._score(by_id.get("code"), 50)
            overall = sec * 0.5 + sent * 0.25 + code * 0.25
            rec = "BUY" if overall >= 75 else "HOLD" if overall >= 50 else "AVOID"
            summary = f"Token Score: {overall:.0f}/100 - {rec}"
            details = {"overall_score": round(overall,1), "security_score": sec, "sentiment_score": sent, "code_score": code,
                       "security_data": by_id.get("security",SubtaskResult("","",False)).data,
                       "sentiment_data": by_id.get("sentiment",SubtaskResult("","",False)).data,
                       "code_data": by_id.get("code",SubtaskResult("","",False)).data}
            recs = [f"{rec} - {'Strong' if overall>=75 else 'Mixed' if overall>=50 else 'Risky'} fundamentals"]
        elif plan.task_type == TaskType.OPPORTUNITY_DISCOVERY:
            summary = f"Opportunity scan: {ok}/{len(result.subtask_results)} agents reported"
            details = {r.subtask_id: r.data for r in result.subtask_results if r.success}
            recs = ["Cross-reference Tavily results with Quill security scores"]
        elif plan.task_type == TaskType.RESEARCH:
            summary = f"Research briefing from {ok} sources"
            details = {r.subtask_id: r.data for r in result.subtask_results if r.success}
            recs = []
        else:
            summary = f"Query processed via {ok} agents"
            details = {r.subtask_id: r.data for r in result.subtask_results if r.success}
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
    s = Synthesizer().synthesize(result)
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
    s = Synthesizer().synthesize(result)
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
    '    for r in api_routers: app.include_router(r, prefix="/api")\n'
    '    for r in hybrid_routers: app.include_router(r, prefix="/hybrid")\n'
    '    app.mount("/mcp", mcp_app)\n'
    '    logger.info("NEXUS ready: /api/* /hybrid/* /mcp")\n'
    '    return app\n'
)
write("src/mcp_server_nexus/app.py", APP_PY)

# ──────────────────────────────────────────────
# tests/__init__.py
# ──────────────────────────────────────────────
write("tests/__init__.py", "")

print("\\nNEXUS: All files generated successfully!")
print("Files created in src/mcp_server_nexus/")
