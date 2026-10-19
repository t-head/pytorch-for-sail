#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="$REPO_ROOT/.ci/docker/requirements-ci.txt"
WANTED=(pytest pytest-xdist pytest-flakefinder pytest-rerunfailures pytest-subtests expecttest hypothesis pyyaml)
SPECS_RAW="$(python - "$REQ_FILE" "${WANTED[@]}" <<'PY'
import re
import sys

req_file, wanted = sys.argv[1], sys.argv[2:]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

found: dict[str, str] = {}
with open(req_file) as f:
    for line in f:
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*((?:==|>=|<=|~=|!=|<|>).+)$", line)
        if m and norm(m.group(1)) not in found:
            found[norm(m.group(1))] = m.group(1) + m.group(2).replace(" ", "")

missing = [p for p in wanted if norm(p) not in found]
if missing:
    sys.exit(f"[deps] Version constraints not found in requirements-ci.txt: {' '.join(missing)}")
print(" ".join(found[norm(p)] for p in wanted))
PY
)"
read -r -a PKGS <<<"$SPECS_RAW"
echo "=== Target versions (from .ci/docker/requirements-ci.txt) ==="
printf '  %s\n' "${PKGS[@]}"

echo "=== Test framework dependency self-check ==="
if python - "${PKGS[@]}" <<'PY'
import re
import sys
from importlib.metadata import PackageNotFoundError, version


def key(v: str) -> tuple[int, ...]:
    # Compare numeric release segments only (7.3.2 -> (7,3,2)); enough for these packages and avoids depending on packaging
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4])


def pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def clause_ok(got: str, op: str, want: str) -> bool:
    if op in ("==", "!=") and want.endswith(".*"):
        prefix = key(want[:-2])
        hit = key(got)[: len(prefix)] == prefix
        return hit if op == "==" else not hit
    a, b = pad(key(got), key(want))
    return {
        "==": a == b,
        "!=": a != b,
        ">=": a >= b,
        "<=": a <= b,
        ">": a > b,
        "<": a < b,
        "~=": a >= b and a[: len(b) - 1] == b[: len(b) - 1],
    }[op]


unsatisfied = []
for spec in sys.argv[1:]:
    m = re.match(r"^([A-Za-z0-9._-]+)(.+)$", spec)
    clauses = [re.match(r"^(==|!=|>=|<=|~=|<|>)(.+)$", c) for c in m.group(2).split(",")] if m else [None]
    if not all(clauses):
        print(f"  {spec} unrecognized version constraint")
        unsatisfied.append(spec)
        continue
    name, constraint = m.groups()
    try:
        got = version(name)
    except PackageNotFoundError:
        print(f"  {name} not installed (requires {constraint})")
        unsatisfied.append(spec)
        continue
    ok = all(clause_ok(got, *c.groups()) for c in clauses)
    print(f"  {name} {got} (requires {constraint}) {'OK' if ok else 'NOT SATISFIED'}")
    if not ok:
        unsatisfied.append(spec)
raise SystemExit(1 if unsatisfied else 0)
PY
then
    echo "[deps] All versions satisfied, skipping install"
    exit 0
fi

source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
ppu_build_pip_candidates
CANDIDATES=("${PIP_CANDIDATES[@]}")

echo "=== pip index connectivity diagnostics ==="
echo "[probe] Proxy environment variables set (names only, values not printed to avoid leaking embedded credentials): $(
    _set_proxy_names=""
    for _pv in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
        [[ -n "${!_pv:-}" ]] && _set_proxy_names="${_set_proxy_names}${_pv} "
    done
    printf '%s' "${_set_proxy_names:-<none>}"
)"
echo "[probe] pip config:"
python -m pip config list 2>&1 | sed 's/^/        /' || true
python - "${CANDIDATES[@]}" <<'PY' || true
import socket
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit

dns_cache: dict[str, str] = {}
for index in sys.argv[1:]:
    host = urlsplit(index).hostname or "?"
    if host not in dns_cache:
        try:
            dns_cache[host] = ", ".join(socket.gethostbyname_ex(host)[2])
        except OSError as exc:
            dns_cache[host] = f"resolution failed ({exc})"
    print(f"[probe] {index}")
    print(f"        dns({host}) = {dns_cache[host]}")
    url = index.rstrip("/") + "/pytest-rerunfailures/"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            print(f"        GET .../pytest-rerunfailures/ = {resp.status}")
    except urllib.error.HTTPError as exc:
        # 404 = the repo/package does not exist in this Artifactory view; 401/403 = authentication required
        print(f"        GET .../pytest-rerunfailures/ = {exc.code} {exc.reason}")
    except Exception as exc:
        print(f"        GET .../pytest-rerunfailures/ = unreachable ({type(exc).__name__}: {exc})")
PY

echo "=== Installing test framework dependencies ==="
for index in "${CANDIDATES[@]}" __pip_default__; do
    pip_args=(--disable-pip-version-check --retries 1 --timeout 20)
    if [[ "${index}" == "__pip_default__" ]]; then
        label="pip default index"
    else
        label="${index}"
        pip_args+=(-i "${index}")
    fi
    echo "[deps] Trying index: ${label}"
    if python -m pip install "${pip_args[@]}" "${PKGS[@]}"; then
        echo "[deps] Install succeeded (index: ${label})"
        exit 0
    fi
    echo "[deps][warn] Index ${label} failed, trying the next one"
done

cat >&2 <<'EOF'
[deps][error] None of the candidate pip indexes could install the test framework dependencies listed above.
Use the output of the "pip index connectivity diagnostics" section above to decide:
  - all 404             -> the Artifactory view inside the pod lacks these pypi repos; ask ops which repo name the pod should use,
                           then change PIP_INDEX in the workflow (or append candidates via PIP_INDEX_FALLBACKS).
  - 401/403             -> the index requires authentication; provide credentials inside the pod (pip.conf / .netrc).
  - DNS resolution fails -> the pod is not attached to a DNS that can resolve the domain.
  - public index unreachable -> the pod has no internet egress and can only use internal indexes.
Fallback: have the image preinstall these packages at the versions in .ci/docker/requirements-ci.txt; this script skips installation when versions are already satisfied.
EOF
exit 1