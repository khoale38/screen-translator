# Screen Translator

Real-time translation of the text on your screen: game dialogue, hard-subbed video, mission objectives, menus. The translation is drawn right over the original text, or added to OBS as live subtitles. It runs **entirely on your computer**: on-device OCR, a local LLM through llama.cpp, no account, no API keys, and the text you read never leaves your computer.

Unlike line-by-line translators, it **keeps context**. Every new line is translated together with the last few lines and their translations, so names, pronouns, gender, tone and sentences split across screens stay consistent.

```
カイ  ユキはまだ帰ってこないの？          → Kai: Hasn't Yuki come home yet?
アオイ 彼女、昨日から連絡がないんだ。      → Aoi: She hasn't been in touch since yesterday.
カイ  まさか、あの森に行ったんじゃ…        → Kai: Could it be... she went to that forest?
アオイ …行ったよ。止めたけど聞かなかった。  → Aoi: ...She went. I tried to stop her, but she wouldn't listen.
```

Translated without context, the third line comes out as *"Could **you** have gone to that forest?"*, and the fourth loses its subjects entirely.

## Install

Download the latest build from the [Releases page](https://github.com/khoale38/screen-translator/releases/latest). The translation model isn't included: the app downloads it from Hugging Face the first time (2.7 GB for the default), after checking there's room on the disk.

### macOS (Apple Silicon, macOS 13 or later)

1. Download `ScreenTranslator-<version>-macos-arm64.dmg`, open it and drag **Screen Translator** into **Applications**.
2. Open the app. It isn't notarized by Apple yet, so macOS blocks the first launch: go to **System Settings → Privacy & Security**, scroll down and click **Open Anyway** next to the Screen Translator message. On macOS 13 and 14 you can instead right-click the app and choose **Open**.
3. When you press **Start**, allow Screen Recording (see [macOS permission](#macos-permission)).

The app lives in the **menu bar** (the **文A** icon), not the Dock. That's what lets its translations appear on top of full-screen apps such as full-screen YouTube or games. Use the icon to bring the window back, start or stop, or quit. Translation runs on the GPU through Metal.

### Windows (10 or 11, 64-bit)

1. Download `ScreenTranslator-<version>-windows-x64-setup.exe` and run it. It installs for your user only and doesn't need admin rights.
2. The installer isn't code-signed yet, so SmartScreen may say *Windows protected your PC*. Click **More info → Run anyway**.
3. Start **Screen Translator** from the Start menu.

There's also a portable `ScreenTranslator-<version>-windows-x64.zip`: extract it anywhere and run `screen-translator.exe`.

The Windows build runs the model on your graphics card through Vulkan (NVIDIA, AMD or Intel; keep the graphics driver up to date). Without a Vulkan driver it falls back to the CPU, which is much slower (a few seconds per line); if that's too slow, pick **Qwen3.5 2B**. It needs a processor with AVX2 (most PCs from 2013 on). The default OCR models are included, so the first run works offline.

### macOS permission

macOS only lets an app see other windows after you allow it in **System Settings → Privacy & Security → Screen & System Audio Recording**. Allow whichever app starts Screen Translator (the Screen Translator app, OBS, or your terminal), then restart that app. Without the permission, capture only sees the wallpaper, and the app shows a message saying so. After updating the app, macOS may ask again; if capture still only sees the wallpaper, remove Screen Translator from that list and add it back.

## Using it

1. Pick **From** and **To**. Under **Model**, press **Download model** once if it asks.
2. In **Watching**, say what's on screen, for example *GTA V gameplay: mission objectives, police chases, street slang*. This is what makes slang and idioms come out right.
3. Choose what to **Translate**:
   - **A selected area**: best for subtitles and dialogue boxes. Click **Select areas**, drag the green box over the text and resize it from its edges. **+ Add area** adds more boxes, for example one for the speaker's name and one for the line; all areas share the same context. Click **Done**.
   - **The whole screen**: translates every piece of text on one screen and repaints it in place in the original's colours. With several monitors, pick one under **Screen**. **Skip text** ignores tiny text such as stream chat, which OCR often misreads.
4. Press **Start**.

**Overlay** sets how translations look for selected areas: **Over the text** or **Below the text** (moved above the area when there's no room below), the text size, and whether to draw a **Dark box** behind it or just outlined text.

### Shortcuts

These work from any app, including full-screen games and video. No extra permission is needed.

| macOS | Windows | |
|---|---|---|
| ⌃⇧T | Ctrl+Shift+T | start or stop translating |
| ⌃⇧H | Ctrl+Shift+H | hide or show the translations, to peek at the original |
| ⌃⇧1 … ⌃⇧9 | Ctrl+Shift+1 … 9 | translate the whole of screen 1–9 |

### Tips for better translations

- **Watching** matters most for slang: without it, *"Lose the cops"* can come out word for word.
- **Glossary** (under **Glossary & more…**): fix names and terms once, for example `ユキ = Yuki` or `Lose the cops = Cắt đuôi cảnh sát`, and they're always used.
- **Select just the dialogue box** rather than the whole screen when you can. It's faster and avoids translating UI noise.
- Press **Forget context** on a scene or chapter change.
- Text that's already in your target language is left alone, and controller buttons such as **L1** are kept as written.

## Use with OBS

1. In OBS, open **Tools → Scripts → +** and load `obs/screen_translator_obs.py`.
   - On macOS and Windows, OBS needs a Python install set under **Tools → Scripts → Python Settings**.
2. If you installed the app in its default location, the script finds it on its own. Otherwise set **Screen Translator command** to the `screen-translator` executable. The app then starts and stops with OBS.
3. In the app window, click **Select areas** and drag the box over the subtitles.
4. In the script, click **Add overlay to current scene**. It adds a 1920×1080 Browser Source called *Live Translation*. With selected areas it shows the translations as subtitles; with **The whole screen** it paints each translation over its original text, so the captured screen should fill the canvas.
5. Optional: add the control panel as a dock via **Docks → Custom Browser Docks**, with URL `http://127.0.0.1:8765/panel`.

On Windows, the translations the app draws on screen are hidden from screen recording, so viewers only see the overlay. Tick **Run without a window** in the script if you'd rather not have the app window at all; then pick the capture area under **Open control panel**.

The script also adds OBS hotkeys (**Settings → Hotkeys**) for *start/stop translating* and *forget context*.

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
| `screen=0` | on | don't show whole-screen translations |

## Models

All models run locally through llama.cpp. Pick one in the app, or point **Custom GGUF file** (under **Glossary & more…**) at any chat model you already have.

| Model | Size | Notes |
|---|---|---|
| Qwen3.5 4B (default) | 2.7 GB | first words in about 0.5 s on an Apple Silicon Mac; good quality |
| Qwen3.5 2B | 2.0 GB | fastest, for 8 GB machines, weaker with slang |
| Qwen3.5 9B | 5.7 GB | best quality, needs 16 GB RAM |
| Gemma 3 4B | 2.5 GB | widest language coverage (Gemma terms of use) |

Models are stored in `~/Library/Application Support/ScreenTranslator/models` on macOS, `%LOCALAPPDATA%\ScreenTranslator\models` on Windows, and `~/.local/share/ScreenTranslator/models` on Linux.

## How it works

1. **Capture**: grabs your selected areas, or the chosen screen, a few times per second, and skips OCR when nothing changed. Its own translations are left out of the capture.
2. **OCR**: Apple Vision on macOS (30 languages including CJK). On Windows and Linux, RapidOCR (PP-OCRv6) reads Chinese, Japanese, English and other Latin-script languages out of the box; Korean, Russian/Ukrainian, Thai, Arabic and Hindi use extra models that download the first time you pick that language. HUD symbols and dot rows are dropped, and misread button icons (`LI`, `Lt`) become `L1`.
3. **Stabilize**: waits until the text stops changing, so typewriter-style dialogue is translated once and complete. Speaker-name labels are kept on their own line, and a line that comes back is shown from cache instead of being translated again.
4. **Translate with context**: a local model gets your glossary, the **Watching** notes and the recent lines as conversation history. Context expires after 5 minutes by default, and **Forget context** clears it. Whole-screen mode translates up to 12 pieces of text per request.
5. **Show**: the translation streams onto the screen as it's written, and over a local WebSocket to the OBS overlay.

Everything listens on `127.0.0.1` only and rejects requests from other websites. The only other network traffic is the model download and, once at startup, a check of this repository's latest release; turn the check off with **Check for updates at startup** in the menu-bar menu.

## From source

Requires Python 3.10+. On macOS, llama.cpp is built with Metal GPU support automatically.

```bash
git clone https://github.com/khoale38/screen-translator.git
cd screen-translator
uv venv && uv pip install -e ".[desktop]"
```

```bash
.venv/bin/screen-translator
```

| Flag | |
|---|---|
| `--headless` | server only, no window |
| `--browser` | open the control panel in your web browser instead of the app window |
| `--port 8765` | change the port |

## Development

```bash
uv pip install pytest pytest-asyncio
.venv/bin/python -m pytest
```

`ST_E2E=1` also runs the slower tests that load a real model (`ST_E2E_MODEL=qwen3.5-4b` to pick which).

To build the macOS app and DMG locally, run `uv pip install pyinstaller` and then `./packaging/build_macos.sh`. In GitHub Actions, pushing a tag such as `v0.1.0-rc1` builds the macOS and Windows packages as downloadable test builds; a tag that exactly matches the version in `pyproject.toml`, such as `v0.1.0`, also publishes them as a GitHub Release.

`examples/dialogue-demo.html` plays a typewriter-style Japanese conversation. Open it in a browser window and select it as an area to try the whole pipeline.

## License

MIT. Models are downloaded from their publishers under their own licenses: Qwen3.5 under Apache-2.0, Gemma 3 under the Gemma Terms of Use.
