from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "screenshots"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
PAGES = {
    "overview": "01_overview.png",
    "models": "02_model_comparison.png",
    "robustness": "03_robustness.png",
    "ablation": "04_ablation.png",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE),
            headless=True,
        )
        page = browser.new_page(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
        )
        errors = []
        page.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )
        for key, filename in PAGES.items():
            page.goto(
                f"http://127.0.0.1:8501/?page={key}",
                wait_until="domcontentloaded",
            )
            page.get_by_role(
                "heading",
                name="Dry Bean Dataset 分类分析与多模型对比",
            ).wait_for(timeout=30_000)
            page.wait_for_timeout(2_500)
            page.screenshot(path=str(OUTPUT / filename), full_page=False)
        browser.close()
        relevant = [
            message
            for message in errors
            if "favicon" not in message.lower()
        ]
        if relevant:
            raise RuntimeError("浏览器控制台错误: " + " | ".join(relevant))


if __name__ == "__main__":
    main()

