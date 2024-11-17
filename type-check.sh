mypy --disallow-untyped-defs -p pyzk && \
black . --exclude=notebooks --exclude=.venv,.lib && \
ruff --target-version=py310 --fix ./pyzk