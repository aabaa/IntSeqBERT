# Presentation build

- `slides.tex`: スライド本体
- `slides.pdf`: 配布用PDF

## 発表時の読み方

スライド内の濃い青字（`#174EA6`）の文を上から順に拾うと、説明がつながる構成です。
太字は重要な語句や数値の強調に使い、読み上げの目印とは分けています。
従来のオレンジ色の強調文字は黒太字に統一し、青字の文中の強調は青太字にしています。
図表のラベル・数式・補足文は、必要に応じて説明を加えるために残しています。
別の読み上げ原稿は使いません。
読み上げ文の一般語彙と構文はCEFR B2程度を目安にし、短く平易な表現を使います。
研究上必要な専門用語は残し、初出時に簡単に説明します。
数値・個数は青字の読み上げ文も含めて算用数字で表し、割合は `%` を使います。
`one stream ... the other` のような指示表現や `zero-shot` などの専門用語は維持します。

## コンパイラ

WSLに **Tectonic 0.17.0** (Linux x86_64 musl) をインストールしています。

- 実行ファイル: `~/.local/bin/tectonic`（PATHに登録済み）
- TeXパッケージのキャッシュ: `~/.cache/tectonic`
- インストール日: 2026年9月23日

9月18日・22日は `/tmp/intseqbert-slides-build/tectonic` を使っていましたが、
一時領域の消失に備えて上記の恒久配置に変更しました。

## ビルド

リポジトリルートから実行します。不足するTeXパッケージは自動取得されるため、
初回や新しいパッケージを使う際にはネットワーク接続が必要です。

```bash
tectonic --version
mkdir -p /tmp/intseqbert-slides-build
tectonic --outdir /tmp/intseqbert-slides-build --keep-logs \
  paper/2026-03/presentation/slides.tex
```

`/tmp/intseqbert-slides-build/slides.log` の警告と生成されたPDFのレイアウトを確認した後、配布版を更新します。
一時領域に置くのはビルド出力のみで、コンパイラとキャッシュはホームディレクトリに残ります。

```bash
cp /tmp/intseqbert-slides-build/slides.pdf paper/2026-03/presentation/slides.pdf
```

## 検証状況

2026年9月23日: `--only-cached` で再ビルドできること、全30ページの表示、
Overfull/Underfull警告がないことを確認済みです。Firaフォント未導入に伴う
代替フォントの使用と、PDFしおりに関する警告は残っています。

## 別のWSL環境へのインストール

```bash
set -e
archive=$(mktemp /tmp/tectonic.XXXXXX.tar.gz)
curl -L --fail --silent --show-error \
  'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz' \
  -o "$archive"
mkdir -p "$HOME/.local/bin"
tar -xzf "$archive" -C "$HOME/.local/bin" tectonic
rm "$archive"
```

`~/.local/bin` がPATHにない環境では、シェルの設定で追加してください。
TeX Liveが別途利用できる環境では、`presentation` ディレクトリで
`pdflatex slides && pdflatex slides` でもビルドできます。
