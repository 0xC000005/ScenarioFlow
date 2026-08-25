FROM databricksruntime/standard:18.3-LTS

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

WORKDIR /opt/scenarioflow

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN uv venv .venv --python /databricks/python3/bin/python \
    && uv export --quiet --frozen --no-dev --no-emit-project --prune torch \
        --extra baseline-classical --extra paper \
        --output-file /tmp/requirements.txt \
    && uv pip install --python .venv/bin/python \
        --requirement /tmp/requirements.txt \
    && uv pip install --python .venv/bin/python \
        torch==2.4.1 --torch-backend cu121 \
    && uv pip install --python .venv/bin/python --no-deps . \
    && rm -f /tmp/requirements.txt

COPY data ./data
COPY reproduction ./reproduction
COPY results/frozen ./results/frozen
COPY scripts ./scripts
COPY artifacts/manifest-v1.tsv artifacts/README.md ./artifacts/
COPY deploy/databricks/run_job.py ./deploy/databricks/run_job.py

RUN chmod a+rx scripts/*.sh deploy/databricks/run_job.py \
    && chmod 0777 /opt/scenarioflow /opt/scenarioflow/artifacts

ENV PATH="/opt/scenarioflow/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    SCENARIOFLOW_ROOT=/opt/scenarioflow \
    SCENARIOFLOW_ARTIFACT_MANIFEST=/opt/scenarioflow/artifacts/manifest-v1.tsv
