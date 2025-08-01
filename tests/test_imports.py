from importlib import import_module
import asyncio
import pytest

def test_import_gabriel():
    assert import_module("gabriel")


def test_viewer_exposed():
    pytest.importorskip("matplotlib")
    gabriel = import_module("gabriel")
    assert callable(gabriel.view_coded_passages)


def test_openai_client_dummy():
    gabriel = import_module("gabriel")
    client = gabriel.core.OpenAIClient(api_key="sk-test")
    async def run():
        return await client.get_all_responses(
            prompts=["hello"],
            identifiers=["1"],
            use_dummy=True,
        )

    df = asyncio.run(run())
    assert not df.empty
