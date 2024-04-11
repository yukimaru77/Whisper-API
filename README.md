## イメージの作成

```
docker build -t whisper-api ./
```

## コンテナの起動

```
docker run -p 8080:8080 --gpus all --name whisper-api-server  whisper-api
```

## その他情報

* モデル:openai/whisper-large-v3"
* エンドポイント: http://localhost:8080/whisper
* WebUI : http://localhost:8080/docs

```python
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=32,
```