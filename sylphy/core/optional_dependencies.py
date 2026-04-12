from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def build_optional_dependency_error(
    *,
    feature: str,
    extra: str,
    packages: Iterable[str] | None = None,
) -> ImportError:
    pkg_list = [pkg for pkg in packages or () if pkg]
    if not pkg_list:
        pkg_text = "optional dependencies"
        install_target = "them"
    elif len(pkg_list) == 1:
        pkg_text = f"optional dependency '{pkg_list[0]}'"
        install_target = "it"
    else:
        pkg_text = "optional dependencies " + ", ".join(f"'{pkg}'" for pkg in pkg_list)
        install_target = "them"

    message = f"{feature} requires {pkg_text}. Install {install_target} with `pip install 'sylphy[{extra}]'`."
    return ImportError(message)


def is_missing_optional_dependency(exc: BaseException, packages: Iterable[str]) -> bool:
    pkg_list = tuple(pkg.lower() for pkg in packages if pkg)
    if not pkg_list:
        return False

    if isinstance(exc, ModuleNotFoundError):
        missing_name = (getattr(exc, "name", "") or "").lower()
        if any(pkg == missing_name for pkg in pkg_list):
            return True

    text = str(exc).lower()
    return any(pkg in text for pkg in pkg_list)


def wrap_optional_dependency_error(
    exc: BaseException,
    *,
    feature: str,
    extra: str,
    packages: Iterable[str],
) -> ImportError | None:
    pkg_list = tuple(packages)
    if not is_missing_optional_dependency(exc, pkg_list):
        return None

    return build_optional_dependency_error(feature=feature, extra=extra, packages=pkg_list)
