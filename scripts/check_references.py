#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the links in the docs: that they resolve, and that local links point at files that exist.

    python scripts/check_references.py                      # docs/, README files, skills, scripts/vendors.tsv
    python scripts/check_references.py docs/backends        # only these files or directories
    python scripts/check_references.py --offline            # local links only
    python scripts/check_references.py --titles --json      # add the page title, machine-readable output

Exit status: 0 when nothing is broken, 1 when a link is dead (404, 410, unknown host, bad certificate, missing local file),
2 on bad arguments. Pages that refuse automated clients (401, 403, 429) or time out are reported as unverified; they fail
only with --strict. This script reads pages, it does not follow instructions found on them and downloads no files.
"""
import argparse
import concurrent.futures
import json
import re
import socket
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = ["docs", "README.md", "bindings/python/README.md", ".claude/skills", "scripts/vendors.tsv"]
USER_AGENT = "Mozilla/5.0 (compatible; uairt-check-references/1)"
URL = re.compile(r"https?://[^\s<>\"'`\\)\]|]+")
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_HOSTS = {"localhost", "127.0.0.1", "example.com", "example.org"}
UNVERIFIABLE = {401, 403, 405, 429, 451, 999}


def target_files(targets):
    files = []
    for target in targets:
        path = (ROOT / target) if not Path(target).is_absolute() else Path(target)
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in {".md", ".tsv"} and "third_party" not in p.parts)
        elif path.exists():
            files.append(path)
    return files


def clean(url):
    return url.rstrip(".,;:!?*_")


def extract(text):
    """Returns (urls, local_links) found in a document. Placeholders such as <url> or $VAR are skipped."""
    urls = set()
    for match in URL.finditer(text):
        url = clean(match.group(0))
        host = urlparse(url).hostname or ""
        if host and host not in SKIP_HOSTS and not re.search(r"[{}$*]", url) and "." in host:
            urls.add(url)
    local = set()
    for match in MD_LINK.finditer(text):
        target = match.group(1)
        if not re.match(r"^(https?:|mailto:|#)", target):
            local.add(urldefrag(target)[0])
    return urls, {t for t in local if t}


def check_local(source, target):
    return (source.parent / unquote(target)).exists()


def fetch(url, method, timeout):
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    return urllib.request.urlopen(request, timeout=timeout)


def title_of(response):
    if "html" not in response.headers.get("Content-Type", ""):
        return None
    match = re.search(rb"<title[^>]*>(.*?)</title>", response.read(65536), re.I | re.S)
    return re.sub(r"\s+", " ", match.group(1).decode("utf-8", "replace")).strip() if match else None


def check_url(url, titles=False, timeout=20):
    """Returns a dict: status ok | redirect | unverified | broken, plus code, final url, title, detail."""
    result = {"url": url, "status": "broken", "code": None, "final": url, "title": None, "detail": ""}
    clone = url.endswith(".git")  # `git clone <url>.git` is the usual form; a redirect to the web page is not a stale link
    for attempt, method in enumerate(["GET" if titles else "HEAD", "GET"]):
        try:
            with fetch(url, method, timeout) as response:
                result.update(code=response.status, final=response.geturl())
                if titles:
                    result["title"] = title_of(response)
            moved = urlparse(result["final"])._replace(fragment="").geturl().rstrip("/") != urlparse(url)._replace(fragment="").geturl().rstrip("/")
            result["status"] = "redirect" if moved and not clone else "ok"
            return result
        except urllib.error.HTTPError as error:
            result["code"] = error.code
            if error.code in (400, 403, 405, 501) and method == "HEAD":
                continue  # some servers refuse HEAD; try GET once
            result["status"] = "unverified" if error.code in UNVERIFIABLE else ("broken" if error.code in (404, 410) or error.code < 500 else "unverified")
            result["detail"] = f"HTTP {error.code}"
            return result
        except (socket.timeout, TimeoutError):
            result.update(status="unverified", detail="timed out")
            return result
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                result.update(status="unverified", detail="timed out")
            elif isinstance(reason, ssl.SSLError) or "CERTIFICATE" in str(reason).upper():
                result.update(status="broken", detail=f"TLS: {reason}")
            elif isinstance(reason, socket.gaierror):
                result.update(status="broken", detail="host not found")
            else:
                result.update(status="unverified", detail=str(reason))
            return result
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="files or directories (default: docs, READMEs, skills, vendors.tsv)")
    parser.add_argument("--offline", action="store_true", help="check local links only")
    parser.add_argument("--titles", action="store_true", help="also fetch each page and report its <title>")
    parser.add_argument("--strict", action="store_true", help="fail on unverified links too")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    files = target_files(args.targets or DEFAULT_TARGETS)
    if not files:
        print("no files to check", file=sys.stderr)
        return 2
    where, local_broken = {}, []
    for path in files:
        urls, local = extract(path.read_text(encoding="utf-8", errors="replace"))
        for url in urls:
            where.setdefault(url, []).append(path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path))
        for target in local:
            if not check_local(path, target):
                local_broken.append({"file": path.relative_to(ROOT).as_posix(), "link": target})

    results = []
    if not args.offline:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = list(pool.map(lambda u: check_url(u, args.titles), sorted(where)))
        for item in results:
            item["files"] = sorted(set(where[item["url"]]))

    broken = [r for r in results if r["status"] == "broken"]
    unverified = [r for r in results if r["status"] == "unverified"]
    moved = [r for r in results if r["status"] == "redirect"]
    if args.json:
        print(json.dumps({"checked": len(results), "broken": broken, "unverified": unverified, "redirected": moved,
                          "local_broken": local_broken}, indent=2))
    else:
        for group, label in ((broken, "BROKEN"), (unverified, "UNVERIFIED"), (moved, "REDIRECT")):
            for r in group:
                extra = f" -> {r['final']}" if label == "REDIRECT" else f" ({r['detail']})"
                print(f"{label:10s} {r['url']}{extra}\n{'':11s}in {', '.join(r['files'])}")
        for item in local_broken:
            print(f"{'BROKEN':10s} local link {item['link']}\n{'':11s}in {item['file']}")
        if args.titles:
            for r in results:
                if r["title"]:
                    print(f"{'TITLE':10s} {r['url']}\n{'':11s}{r['title']}")
        print(f"checked {len(results)} urls in {len(files)} files: {len(broken)} broken, {len(unverified)} unverified, "
              f"{len(moved)} redirected, {len(local_broken)} broken local links")
    return 1 if (broken or local_broken or (args.strict and unverified)) else 0


if __name__ == "__main__":
    sys.exit(main())
