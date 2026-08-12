import json
from collections.abc import Iterator
from pathlib import Path

from tiny_llm.config import DataConfig, DataSourceConfig
from tiny_llm.data import pipeline


def test_default_data_is_wiki_qa_books() -> None:
    config = DataConfig()

    assert [source.name for source in config.sources] == [
        "wikipedia",
        "question-answering",
        "books",
    ]
    assert config.sources[1].format == "question_answer"


def test_legacy_single_source_config_is_migrated() -> None:
    config = DataConfig.model_validate(
        {"dataset": "roneneldan/TinyStories", "max_documents": 500}
    )

    assert len(config.sources) == 1
    assert config.sources[0].dataset == "roneneldan/TinyStories"
    assert config.sources[0].max_documents == 500


def test_prepare_corpus_interleaves_sources(monkeypatch: object, tmp_path: Path) -> None:
    sources = [
        DataSourceConfig(name="wiki", dataset="wiki", max_documents=100),
        DataSourceConfig(name="books", dataset="books", max_documents=100),
    ]

    def fake_source_texts(source: DataSourceConfig, cache_dir: Path) -> Iterator[str]:
        del cache_dir
        for index in range(source.max_documents):
            yield f"{source.name}-{index}"

    monkeypatch.setattr(pipeline, "_source_texts", fake_source_texts)  # type: ignore[attr-defined]
    corpus = pipeline.prepare_corpus(DataConfig(sources=sources), tmp_path)
    rows = [json.loads(line)["text"] for line in corpus.read_text().splitlines()]

    assert rows[:4] == ["wiki-0", "books-0", "wiki-1", "books-1"]
    manifest = json.loads((corpus.parent / "manifest.json").read_text())
    assert manifest["sources"] == {"wiki": 100, "books": 100}


def test_question_answer_rows_are_rendered_for_causal_training() -> None:
    source = DataSourceConfig(
        name="qa", dataset="qa", format="question_answer", max_documents=100
    )
    row = {
        "context": "The sky appears blue.",
        "question": "What color does the sky appear?",
        "answers": {"text": ["blue"]},
    }

    assert pipeline._normalize_row(row, source) == (
        "Context: The sky appears blue.\n"
        "Question: What color does the sky appear?\n"
        "Answer: blue"
    )
