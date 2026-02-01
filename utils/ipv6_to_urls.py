import sys


PORT = 8000
PATH = "/v1/chat/completions"


def main() -> None:
    ips: list[str] = []
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        # allow either newline-separated or comma/space-separated input
        parts = raw.replace(",", " ").split()
        for ip in parts:
            ip = ip.strip()
            if ip:
                ips.append(ip)

    urls = [f"http://[{ip}]:{PORT}{PATH}" for ip in ips]
    sys.stdout.write(",".join(urls))
    if urls:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()

