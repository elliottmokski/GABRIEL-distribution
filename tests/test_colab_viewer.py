import pandas as pd
import pytest


def test_view_coded_passages_colab_runs():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view_coded_passages

    df = pd.DataFrame({"text": ["A snippet"], "cat": [["A snippet"]]})
    # Should not raise when using the lightweight HTML viewer
    view_coded_passages(df, "text", ["cat"], colab=True)


def test_top_level_view_coded_passages_colab_runs():
    pytest.importorskip("IPython")
    pytest.importorskip("aiolimiter")
    pytest.importorskip("openai")
    import gabriel

    df = pd.DataFrame({"text": ["A snippet"], "cat": [["A snippet"]]})
    gabriel.view_coded_passages(df, "text", ["cat"], colab=True)
