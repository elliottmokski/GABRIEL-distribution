import asyncio
import pandas as pd

from gabriel.core.prompt_template import PromptTemplate
from gabriel.utils.teleprompter import Teleprompter
from gabriel.utils import openai_utils
from gabriel.tasks.ratings import Ratings, RatingsConfig
from gabriel.tasks.deidentification import Deidentifier, DeidentifyConfig
from gabriel.tasks.basic_classifier import BasicClassifier, BasicClassifierConfig
from gabriel.tasks.regional import Regional, RegionalConfig
from gabriel.tasks.county_counter import CountyCounter
from gabriel.utils import PromptParaphraser, PromptParaphraserConfig


def test_prompt_template():
    tmpl = PromptTemplate.from_package("ratings_prompt.jinja2")
    text = tmpl.render(attributes=["a"], descriptions=["desc"], passage="x", object_category="obj", attribute_category="att", format="json")
    assert "desc" in text


def test_teleprompter():
    tele = Teleprompter()
    out = tele.generic_elo_prompt(text_circle="a", text_square="b", attributes=["one"], instructions="test")
    assert "test" in out


def test_get_response_dummy():
    responses, _ = asyncio.run(openai_utils.get_response("hi", use_dummy=True))
    assert responses and responses[0].startswith("DUMMY")


def test_get_response_images_dummy():
    responses, _ = asyncio.run(
        openai_utils.get_response("hi", images=["abcd"], use_dummy=True)
    )
    assert responses and responses[0].startswith("DUMMY")


def test_get_all_responses_dummy(tmp_path):
    df = asyncio.run(openai_utils.get_all_responses(
        prompts=["a", "b"],
        identifiers=["1", "2"],
        save_path=str(tmp_path / "out.csv"),
        use_dummy=True,
    ))
    assert len(df) == 2


def test_get_all_responses_images_dummy(tmp_path):
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a"],
            identifiers=["1"],
            prompt_images={"1": ["abcd"]},
            save_path=str(tmp_path / "img.csv"),
            use_dummy=True,
        )
    )
    assert len(df) == 1


def test_ratings_dummy(tmp_path):
    cfg = RatingsConfig(attributes={"helpfulness": ""}, save_dir=str(tmp_path), file_name="ratings.csv", use_dummy=True)
    task = Ratings(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, text_column="text"))
    assert not df.empty
    assert "helpfulness" in df.columns


def test_deidentifier_dummy(tmp_path):
    cfg = DeidentifyConfig(save_path=str(tmp_path/"deid.csv"), use_dummy=True)
    task = Deidentifier(cfg)
    data = pd.DataFrame({"text": ["John went to Paris."]})
    df = asyncio.run(task.run(data, text_column="text"))
    assert "deidentified_text" in df.columns


def test_basic_classifier_dummy(tmp_path):
    cfg = BasicClassifierConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = BasicClassifier(cfg)
    df = pd.DataFrame({"txt": ["a", "b"]})
    res = asyncio.run(task.run(df, text_column="txt"))
    assert "yes" in res.columns


def test_regional_dummy(tmp_path):
    data = pd.DataFrame({"county": ["A", "B"]})
    cfg = RegionalConfig(save_dir=str(tmp_path), use_dummy=True)
    task = Regional(data, "county", topics=["economy"], cfg=cfg)
    df = asyncio.run(task.run())
    assert "economy" in df.columns


def test_county_counter_dummy(tmp_path):
    data = pd.DataFrame({"county": ["A"], "fips": ["00001"]})
    counter = CountyCounter(
        data,
        county_col="county",
        topics=["econ"],
        fips_col="fips",
        save_dir=str(tmp_path),
        use_dummy=True,
        n_elo_rounds=1,
    )
    df = asyncio.run(counter.run())
    assert "econ" in df.columns


def test_prompt_paraphraser_ratings(tmp_path):
    cfg = RatingsConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        file_name="rat.csv",
        use_dummy=True,
    )
    parap_cfg = PromptParaphraserConfig(n_variants=2, save_dir=str(tmp_path / "para"), use_dummy=True)
    paraphraser = PromptParaphraser(parap_cfg)
    data = pd.DataFrame({"txt": ["hello"]})
    df = asyncio.run(paraphraser.run(Ratings, cfg, data, text_column="txt"))
    assert set(df.prompt_variant) == {"baseline", "variant_1", "variant_2"}

