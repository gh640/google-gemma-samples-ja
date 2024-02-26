# Google Gemma サンプル集

Google の Gemma モデルのサンプル集です。

## 前提

- macOS Sonoma (14)
- Python 3.12
- Poetry `>=1.8.0`
- Hugging Face アカウント
- Gemma モデルの利用規約に同意済み

## 確認時の環境

```zsh
sw_vers
```

```text
ProductName:		macOS
ProductVersion:		14.3.1
BuildVersion:		23D60
```

## 使い方

Python 3.12 と Poetry 1.x はインストール済みのものとします。

Poetry で必要な PyPI パッケージをインストールします。

```bash
poetry install --no-root
```

Hugging Face のトークンを取得して環境変数 `HF_TOKEN` にセットします。
トークンの取得がまだの場合は Hugging Face の [Access Tokens ページ](https://huggingface.co/settings/tokens) で生成・取得します。

```bash
export HF_TOKEN='...'
```

Poetry の venv 内でサンプルを実行します。

```bash
poetry run python samples/01/gemma-b2.py
```

## リンク

### Google 公式ページ

- [Gemma: Google introduces new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/)
- [Gemma models overview | Google AI for Developers](https://ai.google.dev/gemma/docs)

### Hugging Face

- [google/gemma-2b · Hugging Face](https://huggingface.co/google/gemma-2b)
- [google/gemma-2b-it · Hugging Face](https://huggingface.co/google/gemma-2b-it)
- [google/gemma-7b · Hugging Face](https://huggingface.co/google/gemma-7b)
- [google/gemma-7b-it · Hugging Face](https://huggingface.co/google/gemma-7b-it)
