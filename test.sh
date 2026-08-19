cd /mnt/share/t00xxxxx/Megatron-LM

git branch --show-current
git rev-parse --short HEAD
grep -nE '^(version *=|__version__)' pyproject.toml setup.py megatron/core/package_info.py 2>/dev/null