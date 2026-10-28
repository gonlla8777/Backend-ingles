import logging
import re
from typing import Optional
import httpx

from app.schemas.dictionary_schemas import WordDefinitionOut, WordMeaningOut

logger = logging.getLogger(__name__)

_POS_MAP = {
    "n": "noun",
    "v": "verb",
    "adj": "adjective",
    "adv": "adverb",
    "u": "general",
}


class DictionaryProxyService:
    """
    Servicio proxy para consultas de diccionario en inglés con caché en memoria.
    Cascada: Free Dictionary API -> Datamuse API -> Fallback vacío.
    Desacopla al frontend de Flutter Web de restricciones de CORS.
    """

    def __init__(self) -> None:
        self._cache: dict[str, WordDefinitionOut] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(4.0, connect=3.0),
                headers={
                    "Accept": "application/json",
                    "User-Agent": "EchoFlow-Backend/1.0",
                },
                follow_redirects=True,
            )
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    @staticmethod
    def clean_word(raw_word: str) -> str:
        """Limpia caracteres de puntuación y signos tipográficos."""
        cleaned = re.sub(r"[^\w\-']", "", raw_word)
        cleaned = re.sub(r"^['\-]+|['\-]+$", "", cleaned)
        return cleaned.strip().lower()

    async def lookup(self, raw_word: str) -> Optional[WordDefinitionOut]:
        word = self.clean_word(raw_word)
        if not word or len(word) < 2:
            return None

        if word in self._cache:
            return self._cache[word]

        client = await self.get_client()

        # 1. Intento primario: Free Dictionary API
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    parsed = self._parse_free_dict(word, data[0])
                    if parsed and parsed.meanings:
                        self._cache[word] = parsed
                        return parsed
        except Exception as exc:
            logger.warning(
                f"[DictionaryProxy] FreeDict fallo para '{word}': {exc}"
            )

        # 2. Intento de respaldo: Datamuse API
        try:
            url = "https://api.datamuse.com/words"
            resp = await client.get(url, params={"sp": word, "md": "dp"})
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    parsed = self._parse_datamuse(word, data)
                    if parsed and parsed.meanings:
                        self._cache[word] = parsed
                        return parsed
        except Exception as exc:
            logger.warning(
                f"[DictionaryProxy] Datamuse fallback fallo para '{word}': {exc}"
            )

        return None

    def _parse_free_dict(
        self, target_word: str, entry: dict
    ) -> Optional[WordDefinitionOut]:
        parsed_word = str(entry.get("word") or target_word)

        # Fonética
        phonetic = entry.get("phonetic")
        if not phonetic and isinstance(entry.get("phonetics"), list):
            for p in entry["phonetics"]:
                if isinstance(p, dict) and p.get("text"):
                    phonetic = str(p["text"])
                    break

        meanings: list[WordMeaningOut] = []
        raw_meanings = entry.get("meanings")
        if isinstance(raw_meanings, list):
            for m in raw_meanings:
                if not isinstance(m, dict):
                    continue
                pos = str(m.get("partOfSpeech") or "general")
                raw_defs = m.get("definitions")
                if isinstance(raw_defs, list):
                    for d in raw_defs:
                        if isinstance(d, dict) and d.get("definition"):
                            meanings.append(
                                WordMeaningOut(
                                    part_of_speech=pos,
                                    definition=str(d["definition"]),
                                    example=str(d.get("example"))
                                    if d.get("example")
                                    else None,
                                )
                            )
                            if len(meanings) >= 3:
                                break
                if len(meanings) >= 4:
                    break

        if not meanings:
            return None

        return WordDefinitionOut(
            word=parsed_word,
            phonetic=phonetic,
            meanings=meanings,
        )

    def _parse_datamuse(
        self, target_word: str, items: list
    ) -> Optional[WordDefinitionOut]:
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("word", "")).lower() != target_word:
                continue

            defs = item.get("defs")
            if not isinstance(defs, list) or not defs:
                continue

            meanings: list[WordMeaningOut] = []
            for d_str in defs:
                if not isinstance(d_str, str):
                    continue
                parts = d_str.split("\t", 1)
                pos = _POS_MAP.get(parts[0], "general") if len(parts) > 1 else "general"
                definition = parts[1] if len(parts) > 1 else parts[0]
                meanings.append(
                    WordMeaningOut(
                        part_of_speech=pos,
                        definition=definition,
                        example=None,
                    )
                )
                if len(meanings) >= 3:
                    break

            if meanings:
                return WordDefinitionOut(
                    word=target_word,
                    phonetic=None,
                    meanings=meanings,
                )

        return None


# Instancia singleton del servicio proxy
dictionary_proxy_service = DictionaryProxyService()

