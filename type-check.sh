mypy --disallow-untyped-defs -p zenopy && \
black . --exclude=notebooks --exclude=.venv,.lib && \
ruff --target-version=py310 --fix ./zenopy