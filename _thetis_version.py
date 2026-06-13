from subprocess import CalledProcessError, check_output


def _current_month_version():
    try:
        head_date = check_output(
            ["git", "show", "-s", "--format=%cI", "HEAD"],
            text=True,
        ).strip()
        year, month = (int(part) for part in head_date[:7].split("-"))
    except (CalledProcessError, OSError, ValueError):
        year, month = 0, 0

    since = f"{year:04d}-{month:02d}-01T00:00:00+00:00"
    try:
        count = check_output(
            ["git", "rev-list", "--count", f"--since={since}", "HEAD"],
            text=True,
        ).strip()
    except (CalledProcessError, OSError):
        count = "0"
    return f"{year}.{month}.dev{count}"


def _next_release_update(version):
    tag = str(version.tag)
    if ".post" in tag:
        base, post = tag.rsplit(".post", 1)
        return f"{base}.post{int(post) + 1}"

    parts = tag.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}.post0"
    raise ValueError(f"Unsupported Thetis release tag {tag}")


def thetis_version_scheme(version):
    if version.exact:
        return str(version.tag)

    branch = version.branch or ""
    if (
        branch == "release"
        or branch.endswith("/release")
        or branch.startswith("release-")
        or branch.startswith("release/")
    ):
        return version.format_next_version(_next_release_update)

    return _current_month_version()
