import inspect

import serve


def test_every_page_is_routed_in_do_get():
    """do_GET routes via a hardcoded if/elif chain, not the PAGES dict — so a new
    PAGES entry silently 404s unless a matching branch is added. Guard both the
    loader path and the build path for every registered page."""
    src = inspect.getsource(serve.Handler.do_GET)
    for page, cfg in serve.PAGES.items():
        assert f'"{cfg["loader"]}"' in src, f"{page}: loader {cfg['loader']} not routed in do_GET"
        assert f'"{cfg["build"]}"' in src, f"{page}: build {cfg['build']} not routed in do_GET"
