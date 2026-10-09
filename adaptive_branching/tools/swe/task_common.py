"""Shared Harbor task policy for SWE datasets."""

PUBLIC_DOCKER_NETWORK = "miles-swe-public"


def public_network_compose(*, hide_r2e_tests: bool = False) -> str:
    tmpfs = "    tmpfs:\n" "      - /r2e_tests:rw,noexec,nosuid,nodev,size=65536\n" if hide_r2e_tests else ""
    return (
        "services:\n"
        "  main:\n"
        f"{tmpfs}"
        "    networks:\n"
        "      - miles-swe-public\n"
        "networks:\n"
        "  miles-swe-public:\n"
        "    external: true\n"
        "    name: ${MILES_SWE_DOCKER_NETWORK:-miles-swe-public}\n"
    )
