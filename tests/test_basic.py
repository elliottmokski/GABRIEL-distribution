import asyncio
import pandas as pd

from gabriel.core.prompt_template import PromptTemplate
from gabriel.utils.teleprompter import Teleprompter
from gabriel.utils import openai_utils
from gabriel.tasks.simple_rating import SimpleRating, RatingConfig
from gabriel.tasks.ratings import Ratings, RatingsConfig
from gabriel.tasks.deidentification import Deidentifier, DeidentifyConfig
from gabriel.tasks.identification import Identification, IdentificationConfig


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


def test_get_all_responses_dummy(tmp_path):
    df = asyncio.run(openai_utils.get_all_responses(
        prompts=["a", "b"],
        identifiers=["1", "2"],
        save_path=str(tmp_path / "out.csv"),
        use_dummy=True,
    ))
    assert len(df) == 2


def test_simple_rating_dummy(tmp_path):
    cfg = RatingConfig(attributes={"helpfulness": ""}, save_path=str(tmp_path/"out.csv"), use_dummy=True)
    task = SimpleRating(cfg)
    df = asyncio.run(task.predict(["hello"]))
    assert not df.empty


def test_ratings_dummy(tmp_path):
    cfg = RatingsConfig(attributes={"helpfulness": ""}, save_path=str(tmp_path/"ratings.csv"), use_dummy=True)
    task = Ratings(cfg)
    df = asyncio.run(task.run(["hello"]))
    assert not df.empty


def test_deidentifier_dummy(tmp_path):
    cfg = DeidentifyConfig(save_path=str(tmp_path/"deid.csv"), use_dummy=True)
    task = Deidentifier(cfg)
    data = pd.DataFrame({"text": ["John went to Paris."]})
    df = asyncio.run(task.run(data, text_column="text"))
    assert "deidentified_text" in df.columns


def test_identification_dummy(tmp_path):
    cfg = IdentificationConfig(classes={"yes": "y"}, save_path=str(tmp_path/"out.csv"), use_dummy=True)
    task = Identification(cfg)
    df = asyncio.run(task.classify(["who"]))
    assert not df.empty
