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


def test_candidate_lemmas():
    lemmas_strangers = dictionary_proxy_service.get_candidate_lemmas("strangers")
    assert "stranger" in lemmas_strangers

    lemmas_running = dictionary_proxy_service.get_candidate_lemmas("running")
    assert "run" in lemmas_running

    lemmas_boxes = dictionary_proxy_service.get_candidate_lemmas("boxes")
    assert "box" in lemmas_boxes

    lemmas_went = dictionary_proxy_service.get_candidate_lemmas("went")
    assert "go" in lemmas_went

    lemmas_carried = dictionary_proxy_service.get_candidate_lemmas("carried")
    assert "carry" in lemmas_carried


def test_dictionary_endpoint_invalid_empty_word():
    client = TestClient(app)
    response = client.get("/api/v1/dictionary/%20%20")
    assert response.status_code == 400


def test_dictionary_endpoint_cached_or_live_success():
    client = TestClient(app)
    # Pre-cargar palabra en caché para prueba determinista y offline
    dictionary_proxy_service._cache["testword"] = WordDefinitionOut(
        word="testword",
        phonetic="/ˈtɛstˌwɜːrd/",
        part_of_speech="noun",
        definition="A word used specifically for testing purposes.",
        translation_es="palabra de prueba",
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
    assert data["part_of_speech"] == "noun"
    assert data["translation_es"] == "palabra de prueba"
    assert len(data["meanings"]) == 1
    assert data["meanings"][0]["part_of_speech"] == "noun"
    assert "testing purposes" in data["meanings"][0]["definition"]


def test_dictionary_endpoint_alias():
    client = TestClient(app)
    # Probar alias /dictionary/{word}
    dictionary_proxy_service._cache["aliasword"] = WordDefinitionOut(
        word="aliasword",
        phonetic=None,
        part_of_speech="noun",
        definition="An alias test word.",
        translation_es="palabra alias",
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
async def test_lookup_lemmatization_fallback():
    # Simulamos que 'stranger' está en caché
    dictionary_proxy_service._cache["stranger"] = WordDefinitionOut(
        word="stranger",
        phonetic="/ˈstreɪndʒər/",
        part_of_speech="noun",
        definition="A person whom one does not know.",
        translation_es="desconocido/a",
        meanings=[
            WordMeaningOut(
                part_of_speech="noun",
                definition="A person whom one does not know.",
                example="Don't talk to strangers.",
            )
        ],
    )

    # Buscamos 'strangers' (plural)
    result = await dictionary_proxy_service.lookup("strangers")
    assert result is not None
    assert result.word == "strangers"
    assert "person whom one does not know" in result.definition
    assert "desconocidos/as" in (result.translation_es or "")


@pytest.mark.asyncio
async def test_deterministic_fallback_guarantee():
    # Una palabra inventada o desconocida que no esté en APIs externas ni lemas
    result = await dictionary_proxy_service.lookup("xyzflurbing")
    assert result is not None
    assert result.word == "xyzflurbing"
    assert len(result.meanings) > 0
    assert result.part_of_speech != ""
    assert result.definition != ""
    assert result.translation_es != ""


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
    parsed = dictionary_proxy_service._parse_free_dict("latency", "latency", raw_entry)
    assert parsed is not None
    assert parsed.word == "latency"
    assert parsed.phonetic == "/ˈleɪtənsi/"
    assert len(parsed.meanings) == 1
    assert parsed.meanings[0].part_of_speech == "noun"
    assert "delay" in parsed.meanings[0].definition
    assert parsed.translation_es == "latencia, retardo"


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
    parsed = dictionary_proxy_service._parse_datamuse("caching", "caching", raw_items)
    assert parsed is not None
    assert parsed.word == "caching"
    assert len(parsed.meanings) == 2
    assert parsed.meanings[0].part_of_speech == "noun"
    assert "storing data" in parsed.meanings[0].definition
    assert parsed.translation_es == "almacenamiento en caché"
