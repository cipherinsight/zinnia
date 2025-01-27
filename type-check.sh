mypy --disallow-untyped-defs -p zinnia && \
black . --exclude=notebooks --exclude=.venv,.lib && \
ruff --target-version=py310 --fix ./zinnia