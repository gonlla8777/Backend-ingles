import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.dictionary_proxy_service import dictionary_proxy_service
from app.schemas.dictionary_schemas import WordDefinitionOut, WordMeaningOut


def test_clean_word_utility():
    assert dictionary_proxy_service.clean_word("Hello,") == "hello"
    assert dictionary_proxy_service.clean_word('"world!"') == "world"
    assert dictionary_proxy_service.clean_word("it's") == "it's"
    assert dictionary_proxy_service.clean_word("—systems—") == "systems"
    assert dictionary_proxy_service.clean_word("..") == ""


def test_dictionary_endpoint_invalid_short_word():
    client = TestClient(app)
    response = client.get("/api/v1/dictionary/a")
    assert response.status_code == 400


def test_dictionary_endpoint_cached_or_live_success():
    client = TestClient(app)
    # Pre-cargar palabra en caché para prueba determinista y offline
    dictionary_proxy_service._cache["testword"] = WordDefinitionOut(
        word="testword",
        phonetic="/ˈtɛstˌwɜːrd/",
        meanings=[
            WordMeaningOut(
                part_of_speech="noun",
                definition="A word used specifically for testing purposes.",
                example="This is a testword example.",
            )
        ],
    )

    response = client.get("/api/v1/dictionary/testword")
    assert response.status_code == 200
    data = response.json()
    assert data["word"] == "testword"
    assert data["phonetic"] == "/ˈtɛstˌwɜːrd/"
    assert len(data["meanings"]) == 1
    assert data["meanings"][0]["part_of_speech"] == "noun"
    assert "testing purposes" in data["meanings"][0]["definition"]


def test_dictionary_endpoint_alias():
    client = TestClient(app)
    # Probar alias /dictionary/{word}
    dictionary_proxy_service._cache["aliasword"] = WordDefinitionOut(
        word="aliasword",
        phonetic=None,
        meanings=[
            WordMeaningOut(
                part_of_speech="noun",
                definition="An alias test word.",
                example=None,
            )
        ],
    )

    response = client.get("/dictionary/aliasword")
    assert response.status_code == 200
    data = response.json()
    assert data["word"] == "aliasword"


@pytest.mark.asyncio
async def test_parse_free_dict_structure():
    raw_entry = {
        "word": "latency",
        "phonetic": "/ˈleɪtənsi/",
        "meanings": [
            {
                "partOfSpeech": "noun",
                "definitions": [
                    {
                        "definition": "The delay before a transfer of data begins.",
                        "example": "High latency degrades real-time streaming.",
                    }
                ],
            }
        ],
    }
    parsed = dictionary_proxy_service._parse_free_dict("latency", raw_entry)
    assert parsed is not None
    assert parsed.word == "latency"
    assert parsed.phonetic == "/ˈleɪtənsi/"
    assert len(parsed.meanings) == 1
    assert parsed.meanings[0].part_of_speech == "noun"
    assert "delay" in parsed.meanings[0].definition


@pytest.mark.asyncio
async def test_parse_datamuse_structure():
    raw_items = [
        {
            "word": "caching",
            "score": 100,
            "defs": [
                "n\tthe process of storing data in a cache",
                "v\tproviding with a cache",
            ],
        }
    ]
    parsed = dictionary_proxy_service._parse_datamuse("caching", raw_items)
    assert parsed is not None
    assert parsed.word == "caching"
    assert len(parsed.meanings) == 2
    assert parsed.meanings[0].part_of_speech == "noun"
    assert "storing data" in parsed.meanings[0].definition

