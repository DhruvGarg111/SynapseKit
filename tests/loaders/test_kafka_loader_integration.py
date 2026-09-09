"""Real KafkaLoader integration test using Testcontainers."""

from __future__ import annotations

import uuid

import pytest

_kafka = pytest.importorskip("kafka")
_kafka_container = pytest.importorskip("testcontainers.kafka")
KafkaContainer = _kafka_container.KafkaContainer

from synapsekit.loaders.kafka import KafkaLoader  # noqa: E402


@pytest.fixture(scope="module")
def kafka_bootstrap_server():
    with KafkaContainer() as container:
        yield container.get_bootstrap_server()


def test_kafka_loader_reads_a_bounded_batch(kafka_bootstrap_server):
    topic = f"synapsekit-loader-{uuid.uuid4().hex}"
    producer = _kafka.KafkaProducer(bootstrap_servers=[kafka_bootstrap_server])
    try:
        producer.send(topic, b"first").get(timeout=30)
        producer.send(topic, b"second").get(timeout=30)
        producer.flush()
    finally:
        producer.close()

    docs = KafkaLoader(
        bootstrap_servers=[kafka_bootstrap_server],
        topic=topic,
        max_messages=1,
        timeout_ms=5000,
        group_id=f"synapsekit-loader-test-{uuid.uuid4().hex}",
    ).load()

    assert len(docs) == 1
    assert docs[0].text == "first"
    assert docs[0].metadata["topic"] == topic
