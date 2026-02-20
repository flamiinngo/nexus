FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy the 4 files
COPY pyproject.toml ./
COPY README.md ./
COPY tool_pricing.yaml ./
COPY setup_project.py ./

# Generate the entire project
RUN python setup_project.py

# Install dependencies
RUN uv sync --no-dev

# Expose port
EXPOSE 8200

# Run NEXUS
CMD ["uv", "run", "python", "-m", "mcp_server_nexus"]
