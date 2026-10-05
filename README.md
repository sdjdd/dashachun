# xiaozhi-server-rs

A Rust reimplementation of the server for [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32), a voice-assistant firmware for ESP32 devices.

It is protocol-compatible with the firmware: it serves the OTA config, accepts the
websocket connection, and handles the voice pipeline (VAD → ASR → LLM → TTS →
Opus audio) plus device activation/binding and user accounts.

The project is under active development; the API and internals may change.
