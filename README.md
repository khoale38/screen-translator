# Screen Translator

Real-time translation of whatever text is on your screen (game dialogue, hard-subbed video, livestream chat, UI) shown as live subtitles in OBS or a floating window. It runs **entirely on your computer**: on-device OCR, a local LLM through llama.cpp, no API keys and no cloud calls.

Unlike line-by-line translators, it **keeps context**. Every new line is translated together with the last few lines and their translations, so names, pronouns, gender, tone and sentences split across screens stay consistent.

```
カイ  ユキはまだ帰ってこないの？          → Kai: Hasn't Yuki come home yet?
アオイ 彼女、昨日から連絡がないんだ。      → Aoi: She hasn't been in touch since yesterday.
カイ  まさか、あの森に行ったんじゃ…        → Kai: Could it be... she went to that forest?
アオイ …行ったよ。止めたけど聞かなかった。  → Aoi: ...She went. I tried to stop her, but she wouldn't listen.
```

Translated without context, the third line comes out as *"Could **you** have gone to that forest?"*, and the fourth loses its subjects entirely.

## How it works

1. **Capture**: grabs the screen area you select a few times per second, and skips OCR when nothing changed.
2. **OCR**: Apple Vision on macOS (fast and accurate, 30 languages including CJK). On Windows and Linux, RapidOCR (PP-OCRv6) reads Chinese, Japanese, English and other Latin-script languages out of the box; Korean, Russian/Ukrainian, Thai, Arabic and Hindi use extra models that download the first time you pick that language. Vietnamese text sometimes loses stacked accents on Windows.
3. **Stabilize**: waits until the text stops changing, so typewriter-style dialogue is translated once and complete. Speaker-name labels are kept on their own line.
4. **Translate with context**: a local model (Qwen3.5 by default) gets your glossary, background notes and the recent lines as conversation history. Context expires after a configurable time (5 minutes by default), and you can clear it at any time, for example on a scene change.
5. **Show**: tokens stream over a local WebSocket to the OBS overlay, the control panel and the optional floating subtitle window.

Everything listens on `127.0.0.1` only and rejects requests from other websites.

## Install

Download the latest build from the [Releases page](https://github.com/khoale38/screen-translator/releases/latest). Models aren't included; the app downloads the one you pick on first use.

### macOS (Apple Silicon, macOS 13 or later)

1. Download `ScreenTranslator-<version>-macos-arm64.dmg`, open it and drag **Screen Translator** into **Applications**.
2. Open the app. It isn't notarized by Apple yet, so macOS blocks the first launch: go to **System Settings → Privacy & Security**, scroll down and click **Open Anyway** next to the Screen Translator message. On macOS 13 and 14 you can instead right-click the app and choose **Open**.
3. When you press **Start**, allow Screen Recording (see [macOS permission](#macos-permission)).

Translation runs on the GPU through Metal.

### Windows (10 or 11, 64-bit)

1. Download `ScreenTranslator-<version>-windows-x64-setup.exe` and run it. It installs for your user only and doesn't need admin rights.
2. The installer isn't code-signed yet, so SmartScreen may say *Windows protected your PC*. Click **More info → Run anyway**.
3. Start **Screen Translator** from the Start menu.

There's also a portable `ScreenTranslator-<version>-windows-x64.zip`: extract it anywhere and run `screen-translator.exe`.

The Windows build runs the model on your graphics card through Vulkan (NVIDIA, AMD or Intel; keep the graphics driver up to date). Without a Vulkan driver it falls back to the CPU, which is several times slower; if the 4B model is too slow there, pick **Qwen3.5 2B**. The default OCR models are included, so the first run works offline.

### From source

Requires Python 3.10+. On macOS, llama.cpp is built with Metal GPU support automatically.

```bash
git clone https://github.com/khoale38/screen-translator.git
cd screen-translator
uv venv && uv pip install -e ".[desktop]"
```

```bash
.venv/bin/screen-translator
```

The control panel opens in its own window. Pick a model and press **Download** (one time, from Hugging Face, checksum-verified). Then choose your languages, select the screen area and press **Start**.

| Flag | |
|---|---|
| `--headless` | server only (this is what the OBS script uses) |
| `--browser` | open the control panel in your web browser instead of a window |
| `--port 8765` | change the port |

### macOS permission

macOS only lets an app see other windows after you allow it in **System Settings → Privacy & Security → Screen & System Audio Recording**. Allow whichever app starts Screen Translator (the Screen Translator app, OBS, or your terminal), then restart that app. Without the permission, capture only sees the wallpaper, and the app shows a message saying so. After updating the app, macOS may ask again; if capture still only sees the wallpaper, remove Screen Translator from that list and add it back.

## Use with OBS

1. In OBS, open **Tools → Scripts → +** and load `obs/screen_translator_obs.py`.
   - On macOS and Windows, OBS needs a Python install set under **Tools → Scripts → Python Settings**.
2. If you installed the app in its default location, the script finds it on its own. Otherwise set **Screen Translator command** to the `screen-translator` executable, for example `…/screen-translator/.venv/bin/screen-translator`. The app then starts and stops with OBS.
3. Click **Open control panel**, then pick the languages and the capture area.
4. Click **Add overlay to current scene**. It adds a 1920×1080 Browser Source called *Live Translation*.
5. Optional: add the control panel as a dock via **Docks → Custom Browser Docks**, with URL `http://127.0.0.1:8765/panel`.

The script also adds hotkeys (**Settings → Hotkeys**) for *start/stop translating* and *forget context*.

### Overlay options

Add any of these to the Browser Source URL, `http://127.0.0.1:8765/overlay?…`:

| Option | Default | |
|---|---|---|
| `lines` | `2` | how many recent lines to show |
| `size` | `40` | font size in px |
| `source=1` | off | show the original text above the translation |
| `position=top` | bottom | stick subtitles to the top |
| `align=left` | center | left-align |
| `bg` | `0.55` | background opacity, 0–1 |
| `color` | `ffffff` | text colour (hex) |
| `hold` | `0` | hide the subtitles after N seconds (0 means keep them) |

## Models

All models run locally through llama.cpp. Download them from the control panel, or point **Custom GGUF file** at any chat model you already have.

| Model | Size | Notes |
|---|---|---|
| Qwen3.5 4B (default) | 2.7 GB | about 1 s per line on an M1 Pro, good quality |
| Qwen3.5 2B | 2.0 GB | fastest, for 8 GB machines |
| Qwen3.5 9B | 5.7 GB | best quality, needs 16 GB RAM |
| Gemma 3 4B | 2.5 GB | widest language coverage (Gemma terms of use) |

Models are stored in `~/Library/Application Support/ScreenTranslator/models` on macOS, `%LOCALAPPDATA%\ScreenTranslator\models` on Windows, and `~/.local/share/ScreenTranslator/models` on Linux.

## Tips for better translations

- **Glossary**: fix names and terms once (`ユキ = Yuki`), and they stay consistent.
- **Background notes**: one or two sentences about the game or show, for example "Aria is a girl, Kael is her older brother. Casual tone." These help with pronouns and register.
- **Select only the dialogue box**, not the whole screen. It's faster and avoids translating UI noise.
- Press **Forget context** on a scene or chapter change.

## Development

```bash
uv pip install pytest pytest-asyncio
.venv/bin/python -m pytest
```

To build the macOS app and DMG locally, run `uv pip install pyinstaller` and then `./packaging/build_macos.sh`. Pushing a `v*` tag that matches the version in `pyproject.toml` builds macOS and Windows packages in GitHub Actions and attaches them to a GitHub Release.

`examples/dialogue-demo.html` plays a typewriter-style Japanese conversation. Open it in a browser window and point the capture area at it to try the whole pipeline.

## License

MIT. Models are downloaded from their publishers under their own licenses: Qwen3.5 under Apache-2.0, Gemma 3 under the Gemma Terms of Use.
