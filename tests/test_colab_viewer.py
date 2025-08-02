import pandas as pd
import pytest

from gabriel.utils import view_coded_passages


def test_view_coded_passages_colab_runs():
    pytest.importorskip("IPython")

    df = pd.DataFrame({"text": ["A snippet"], "cat": [["A snippet"]]})
    # Should not raise when using the lightweight HTML viewer
    view_coded_passages(df, "text", ["cat"], colab=True)
