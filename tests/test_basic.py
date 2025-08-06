import asyncio
import pandas as pd

from gabriel.core.prompt_template import PromptTemplate
from gabriel.utils.teleprompter import Teleprompter
from gabriel.utils import openai_utils
from gabriel.tasks.rate import Rate, RateConfig
from gabriel.tasks.deidentify import Deidentifier, DeidentifyConfig
from gabriel.tasks.classify import Classify, ClassifyConfig
from gabriel.tasks.regional import Regional, RegionalConfig
from gabriel.tasks.county_counter import CountyCounter
from gabriel.utils import PromptParaphraser, PromptParaphraserConfig
import gabriel


def test_prompt_template():
    tmpl = PromptTemplate.from_package("ratings_prompt.jinja2")
    text = tmpl.render(attributes=["a"], descriptions=["desc"], passage="x", object_category="obj", attribute_category="att", format="json")
    assert "desc" in text


def test_ratings_default_scale_prompt():
    tmpl = PromptTemplate.from_package("ratings_prompt.jinja2")
    rendered = tmpl.render(text="x", attributes=["clarity"], scale=None)
    assert "All ratings are on a scale" in rendered


def test_shuffled_dict_rendering():
    tmpl = PromptTemplate.from_package("classification_prompt.jinja2")
    rendered = tmpl.render(text="x", attributes={"clarity": "Is the text clear?"})
    assert "OrderedDict" not in rendered
    assert "{" in rendered and "}" in rendered


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
    assert set(["Successful", "Error Log"]).issubset(df.columns)
    assert df["Successful"].all()


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
    cfg = RateConfig(attributes={"helpfulness": ""}, save_dir=str(tmp_path), file_name="ratings.csv", use_dummy=True)
    task = Rate(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, text_column="text"))
    assert not df.empty
    assert "helpfulness" in df.columns


def test_ratings_multirun(tmp_path):
    cfg = RateConfig(attributes={"helpfulness": ""}, save_dir=str(tmp_path), file_name="ratings.csv", use_dummy=True, n_runs=2)
    task = Rate(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, text_column="text"))
    assert "helpfulness" in df.columns
    disagg = pd.read_csv(tmp_path / "ratings_full_disaggregated.csv", index_col=[0, 1])
    assert set(disagg.index.names) == {"text", "run"}


def test_deidentifier_dummy(tmp_path):
    cfg = DeidentifyConfig(save_dir=str(tmp_path), file_name="deid.csv", use_dummy=True)
    task = Deidentifier(cfg)
    data = pd.DataFrame({"text": ["John went to Paris."]})
    df = asyncio.run(task.run(data, text_column="text"))
    assert "deidentified_text" in df.columns


def test_classification_dummy(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = Classify(cfg)
    df = pd.DataFrame({"txt": ["a", "b"]})
    res = asyncio.run(task.run(df, text_column="txt"))
    assert "yes" in res.columns


def test_classification_multirun(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True, n_runs=2)
    task = Classify(cfg)
    df = pd.DataFrame({"txt": ["a"]})
    res = asyncio.run(task.run(df, text_column="txt"))
    assert "yes" in res.columns
    disagg = pd.read_csv(tmp_path / "classify_responses_full_disaggregated.csv", index_col=[0, 1])
    assert set(disagg.index.names) == {"text", "run"}


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
    cfg = RateConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        file_name="rat.csv",
        use_dummy=True,
    )
    parap_cfg = PromptParaphraserConfig(n_variants=2, save_dir=str(tmp_path / "para"), use_dummy=True)
    paraphraser = PromptParaphraser(parap_cfg)
    data = pd.DataFrame({"txt": ["hello"]})
    df = asyncio.run(paraphraser.run(Rate, cfg, data, text_column="txt"))
    assert set(df.prompt_variant) == {"baseline", "variant_1", "variant_2"}


def test_api_wrappers(tmp_path):
    df = pd.DataFrame({"txt": ["hello"]})
    rated = asyncio.run(
        gabriel.rate(
            df,
            "txt",
            attributes={"clarity": ""},
            save_dir=str(tmp_path / "rate"),
            use_dummy=True,
        )
    )
    assert "clarity" in rated.columns

    classified = asyncio.run(
        gabriel.classify(
            df,
            "txt",
            labels={"yes": ""},
            save_dir=str(tmp_path / "cls"),
            use_dummy=True,
        )
    )
    assert "yes" in classified.columns

    deidentified = asyncio.run(
        gabriel.deidentify(
            df,
            "txt",
            save_dir=str(tmp_path / "deid"),
            use_dummy=True,
        )
    )
    assert "deidentified_text" in deidentified.columns

    custom = asyncio.run(
        gabriel.custom_prompt(
            prompts=["hello"],
            identifiers=["1"],
            save_dir=str(tmp_path / "cust"),
            file_name="out.csv",
            use_dummy=True,
        )
    )
    assert len(custom) == 1

