# Screen Translator

Real-time translation of the text on your screen: game dialogue, hard-subbed video, mission objectives, menus. It can also listen to what's being said in a game or video and subtitle it. The translation is drawn right over the original text, or added to OBS as live subtitles. It runs **entirely on your computer**: on-device OCR, a local LLM through llama.cpp, no account, no API keys, and the text you read never leaves your computer.

Unlike line-by-line translators, it **keeps context**. Every new line is translated together with the last few lines and their translations, so names, pronouns, gender, tone and sentences split across screens stay consistent.

```
カイ  ユキはまだ帰ってこないの？          → Kai: Hasn't Yuki come home yet?
アオイ 彼女、昨日から連絡がないんだ。      → Aoi: She hasn't been in touch since yesterday.
カイ  まさか、あの森に行ったんじゃ…        → Kai: Could it be... she went to that forest?
アオイ …行ったよ。止めたけど聞かなかった。  → Aoi: ...She went. I tried to stop her, but she wouldn't listen.
```

Translated without context, the third line comes out as *"Could **you** have gone to that forest?"*, and the fourth loses its subjects entirely.

## Install

Screen Translator is released for **Windows 10 and 11 (64-bit)**. Download the latest build from the [Releases page](https://github.com/khoale38/screen-translator/releases/latest). The translation model isn't included: the app downloads it from Hugging Face the first time (2.7 GB for the default), after checking there's room on the disk.

1. Download `ScreenTranslator-<version>-windows-x64-setup.exe` and run it. It installs for your user only and doesn't need admin rights.
2. The installer isn't code-signed yet, so SmartScreen may say *Windows protected your PC*. Click **More info → Run anyway**.
3. Start **Screen Translator** from the Start menu.

There's also a portable `ScreenTranslator-<version>-windows-x64.zip`: extract it anywhere and run `screen-translator.exe`.

The app also puts a **文A** icon in the notification area of the taskbar. Use it to bring the window back, start or stop, or quit.

It runs the model on your graphics card through Vulkan and reads text on it through DirectML (NVIDIA, AMD or Intel; keep the graphics driver up to date). Without a working driver both fall back to the CPU, which is much slower (a few seconds per line); if that's too slow, pick **Qwen3.5 2B**. While it runs, the bottom of the window shows which device the model and the text reading are on and how long each step takes. It needs a processor with AVX2 (most PCs from 2013 on). The default OCR models are included, so the first run works offline.

## Using it

1. Pick **From** and **To**. Under **Model**, press **Download model** once if it asks.
2. In **Watching**, say what's on screen, for example *GTA V gameplay: mission objectives, police chases, street slang*. This is what makes slang and idioms come out right.
3. Choose what to **Translate**:
   - **A selected area**: best for subtitles and dialogue boxes. Click **Select areas**, drag the green box over the text and resize it from its edges. **+ Add area** adds more boxes, for example one for the speaker's name and one for the line; all areas share the same context. Click **Done**.
   - **The whole screen**: translates every piece of text on one screen and repaints it in place in the original's colours. With several monitors, pick one under **Screen**. **Skip text** ignores tiny text such as stream chat, which OCR often misreads.
4. Optionally tick **Also translate what's said** to subtitle speech too, for voice acting, streams and videos without subtitles. It listens to whatever the PC is playing, alongside the on-screen text, and shows spoken lines at the bottom of the screen. It understands Japanese, Chinese, Korean, Cantonese and English. Press **Download speech recognition** once (0.24 GB). A live caption appears while someone is talking, and the finished line usually lands within a second after they stop. Speech gets the model first, so on-screen text waits a moment while someone talks. Lines already spoken in your target language are shown as said.
5. Optionally tick **Read them aloud too (dubbing)** to hear each spoken line in your language. It uses the voices installed in Windows (add more under **Settings → Time & language → Speech**). **Match the speaker** picks a male or female voice from how the speaker sounds, sets it higher or lower to match, and follows the emotion it hears: excited lines come out quicker and higher, sad ones slower and softer, angry ones louder. You can also pick an accent, such as UK or US English, or one particular voice. While a line is read, the game's own sound can be turned down, kept, or muted. The dub starts about a second after the original line ends, and it isn't picked up as new speech.
6. Press **Start**.

**Overlay** sets how translations look for selected areas: **Over the text** or **Below the text** (moved above the area when there's no room below), the text size, and whether to draw a **Dark box** behind it or just outlined text.

### Shortcuts

These work from any app, including full-screen games and video.

| Shortcut | |
|---|---|
| Ctrl+Shift+T | start or stop translating |
| Ctrl+Shift+H | hide or show the translations, to peek at the original |
| Ctrl+Shift+1 … 9 | translate the whole of screen 1–9 |

### Tips for better translations

- **Watching** matters most for slang: without it, *"Lose the cops"* can come out word for word.
- **Glossary** (under **Glossary & more…**): fix names and terms once, for example `ユキ = Yuki` or `Lose the cops = Cắt đuôi cảnh sát`, and they're always used.
- **Select just the dialogue box** rather than the whole screen when you can. It's faster and avoids translating UI noise.
- Press **Forget context** on a scene or chapter change.
- Text that's already in your target language is left alone, and controller buttons such as **L1** are kept as written.

## Use with OBS

1. In OBS, open **Tools → Scripts → +** and load `obs/screen_translator_obs.py`.
   - OBS needs a Python install set under **Tools → Scripts → Python Settings**.
2. If you installed the app in its default location, the script finds it on its own. Otherwise set **Screen Translator command** to the `screen-translator` executable. The app then starts and stops with OBS.
3. In the app window, click **Select areas** and drag the box over the subtitles.
4. In the script, click **Add overlay to current scene**. It adds a 1920×1080 Browser Source called *Live Translation*. With selected areas it shows the translations as subtitles; with **The whole screen** it paints each translation over its original text, so the captured screen should fill the canvas.
5. Optional: add the control panel as a dock via **Docks → Custom Browser Docks**, with URL `http://127.0.0.1:8765/panel`.

The translations the app draws on screen are hidden from screen recording, so viewers only see the overlay. Tick **Run without a window** in the script if you'd rather not have the app window at all; then pick the capture area under **Open control panel**.

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
| Qwen3.5 4B (default) | 2.7 GB | good quality, the best balance of speed and accuracy |
| Qwen3.5 2B | 2.0 GB | fastest, for 8 GB machines, weaker with slang |
| Qwen3.5 9B | 5.7 GB | best quality, needs 16 GB RAM |
| Gemma 3 4B | 2.5 GB | widest language coverage (Gemma terms of use) |

Models are stored in the `models` folder inside the folder you installed to (or unzipped the portable version to), and uninstalling removes them. If that folder is read-only, such as an all-users install under Program Files, they go to `%LOCALAPPDATA%\ScreenTranslator\models` instead. Models downloaded by an earlier version are moved over the first time you start.

## How it works

1. **Capture**: grabs your selected areas, or the chosen screen, a few times per second, and skips OCR when nothing changed. Its own translations are left out of the capture. With **Also translate what's said** it also records the PC's sound output; Silero VAD finds where each sentence starts and stops, and SenseVoice-Small writes it down in under 0.1 s.
2. **OCR**: RapidOCR (PP-OCRv6) reads Chinese, Japanese, English and other Latin-script languages out of the box; Korean, Russian/Ukrainian, Thai, Arabic and Hindi use extra models that download the first time you pick that language. HUD symbols and dot rows are dropped, and misread button icons (`LI`, `Lt`) become `L1`.
3. **Stabilize**: waits until the text stops changing, so typewriter-style dialogue is translated once and complete. Speaker-name labels are kept on their own line, and a line that comes back is shown from cache instead of being translated again.
4. **Translate with context**: a local model gets your glossary, the **Watching** notes and the recent lines as conversation history. Context expires after 5 minutes by default, and **Forget context** clears it. Whole-screen mode translates up to 12 pieces of text at once, side by side on the graphics card, so a screen takes about as long as its longest piece and each piece appears as soon as it's done. For a whole screen in under a second on a mid-range card such as an RTX 2060, pick **Qwen3.5 2B**; the 4B model takes one to two seconds but handles slang better.
5. **Show**: the translation streams onto the screen as it's written, and over a local WebSocket to the OBS overlay.

Everything listens on `127.0.0.1` only and rejects requests from other websites. The only other network traffic is the model download and, once at startup, a check of this repository's latest release; turn the check off with **Check for updates at startup** in the 文A icon's menu.

## From source

Requires Python 3.10+. Running from source also works on macOS (Apple Silicon, macOS 13 or later), which isn't released but is used for development; see [macOS](#macos).

```bash
git clone https://github.com/khoale38/screen-translator.git
cd screen-translator
uv venv && uv pip install -e ".[desktop]"
```

```bash
.venv\Scripts\screen-translator
```

(`.venv/bin/screen-translator` on macOS.)

| Flag | |
|---|---|
| `--headless` | server only, no window |
| `--browser` | open the control panel in your web browser instead of the app window |
| `--port 8765` | change the port |

## Development

```bash
uv pip install pytest pytest-asyncio
.venv\Scripts\python -m pytest
```

`ST_E2E=1` also runs the slower tests that load a real model (`ST_E2E_MODEL=qwen3.5-4b` to pick which).

In GitHub Actions, pushing a tag builds the Windows installer and zip, plus a macOS DMG as a test download on the run; the DMG is never published. A tag matching the version in `pyproject.toml`, such as `v0.1.0`, publishes the Windows files as a GitHub Release. A test tag such as `v0.1.0-rc1` publishes them as a prerelease, which the app's update check ignores.

`examples/dialogue-demo.html` plays a typewriter-style Japanese conversation. Open it in a browser window and select it as an area to try the whole pipeline.

### macOS

macOS isn't released, but the app runs there from source or as the test DMG from a tag build (`./packaging/build_macos.sh` builds it locally after `uv pip install pyinstaller`). Differences from Windows:

- OCR uses Apple Vision (30 languages including CJK) and the model runs on the GPU through Metal; llama.cpp is built with Metal support automatically.
- The app lives in the **menu bar**, not the Dock, which lets its translations appear over full-screen apps.
- Shortcuts use ⌃⇧ (Control+Shift) instead of Ctrl+Shift.
- macOS only lets an app see other windows after you allow it in **System Settings → Privacy & Security → Screen & System Audio Recording**. Allow whichever app starts Screen Translator (the app, OBS, or your terminal), then restart it. Without it, capture only sees the wallpaper. The DMG isn't notarized, so its first launch needs **Open Anyway** under **Privacy & Security**.
- Settings and models are in `~/Library/Application Support/ScreenTranslator`.

## License

MIT. Models are downloaded from their publishers under their own licenses: Qwen3.5 under Apache-2.0, Gemma 3 under the Gemma Terms of Use, SenseVoice-Small under the FunASR model license and Silero VAD under MIT.
