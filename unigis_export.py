
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

LOGIN_URL = "https://cloud.unigis.com/AGUNSA/Login/"
DEFAULT_URL = "https://cloud.unigis.com/AGUNSA/default.aspx"
TRACKING_HOME_URL = "https://cloud.unigis.com/AGUNSA/tracking/home/"


def open_file_default_app(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            import subprocess

            subprocess.Popen(["open", str(path)])
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


BASE_DIR = app_dir()
RES_DIR = resource_dir()
CONFIG_FILE = BASE_DIR / "unigis_config.json"
BUNDLE_MARKER = BASE_DIR / "_internal"
DOWNLOAD_DIR_DEFAULT = BASE_DIR / "descargas_unigis"

IMAGES_DIR = RES_DIR / "IMAGENES"
WELCOME_IMG = IMAGES_DIR / "Aviso Bienvenido.png"
SETTINGS_IMG = IMAGES_DIR / "Seleccionar Columnas y Exportar Tabla.png"
EXPORT_IMG = IMAGES_DIR / "Seleccionar Columnas y Exportar Tabla.png"

VIEWPORT = {"width": 1267, "height": 926}
STEP_WAIT_MS = 8000
REFRESH_EVERY_SECONDS = 60 * 60
FIXED_EXPORT_NAME = "unigis_export.xlsx"

KEEP_COLUMNS = {"Dominio", "Transporte", "Tipo", "Latitud", "Longitud"}

ALL_COLUMNS = [
    "ID",
    "Etiqueta",
    "Dominio",
    "Transporte",
    "Reportes",
    "Contacto",
    "Tipo",
    "Fecha Evento",
    "Antigüedad",
    "Fecha Reportado",
    "Velocidad",
    "Latitud",
    "Longitud",
    "Referencia",
    "Detenido",
    "Tiempo Detenido",
    "Fecha Inicio Detención",
    "Última Entrada Zona",
    "Última Salida Zona",
    "Tiempo en Zona",
    "Zona",
    "Punto Cercano",
    "En Punto",
    "Distancia Punto",
    "Ubicar",
    "Estado Motor",
    "Patente",
    "Chofer",
    "Nombre",
    "Patrón",
]


@dataclass
class AppConfig:
    username: str = ""
    password: str = ""
    save_credentials: bool = False
    download_dir: str = str(DOWNLOAD_DIR_DEFAULT)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _normalize_download_dir(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def load_config() -> AppConfig:
    if not CONFIG_FILE.exists():
        return AppConfig()

    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return AppConfig(
            username=str(data.get("username", "")),
            password=str(data.get("password", "")),
            save_credentials=bool(data.get("save_credentials", False)),
            download_dir=str(data.get("download_dir", str(DOWNLOAD_DIR_DEFAULT))),
        )
    except Exception:
        return AppConfig()


def save_config(cfg: AppConfig) -> None:
    CONFIG_FILE.write_text(
        json.dumps(
            {
                "username": cfg.username if cfg.save_credentials else "",
                "password": cfg.password if cfg.save_credentials else "",
                "save_credentials": cfg.save_credentials,
                "download_dir": cfg.download_dir,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def persist_config(username: str, password: str, save_credentials: bool, download_dir: Path) -> None:
    save_config(
        AppConfig(
            username=username if save_credentials else "",
            password=password if save_credentials else "",
            save_credentials=save_credentials,
            download_dir=str(download_dir),
        )
    )


def verify_runtime_layout() -> None:
    if not getattr(sys, "frozen", False):
        return

    missing = []
    if not BUNDLE_MARKER.exists():
        missing.append("_internal")
    if not IMAGES_DIR.exists():
        missing.append("IMAGENES")

    if missing:
        raise RuntimeError(
            "La carpeta del programa está incompleta. "
            f"Faltan: {', '.join(missing)}. "
            "Copia la carpeta completa generada por PyInstaller, no solo el .exe."
        )


def _click_first(page, selectors: Iterable[str], timeout: int = 5000) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.click(timeout=timeout)
                return True
        except Exception:
            pass
    return False


def _wait_ready(page, extra_ms: int = 0) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(STEP_WAIT_MS + extra_ms)


def _read_image(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _screenshot_bgr(page) -> np.ndarray:
    raw = page.screenshot(type="png", full_page=False)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("No se pudo leer la captura del navegador.")
    return img


def _template_click(
    page,
    template_path: Path,
    *,
    region: Optional[Tuple[int, int, int, int]] = None,
    threshold: float = 0.62,
) -> bool:
    template = _read_image(template_path)
    if template is None:
        return False

    screen = _screenshot_bgr(page)
    off_x = 0
    off_y = 0

    if region is not None:
        x, y, w, h = region
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        screen = screen[y : y + h, x : x + w]
        off_x = x
        off_y = y

    if screen.size == 0 or template.size == 0:
        return False

    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_score = -1.0
    best_loc = None
    best_size = None

    for scale in (1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15):
        resized = cv2.resize(
            template_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )

        if (
            resized.shape[0] < 4
            or resized.shape[1] < 4
            or resized.shape[0] > screen_gray.shape[0]
            or resized.shape[1] > screen_gray.shape[1]
        ):
            continue

        result = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_size = (resized.shape[1], resized.shape[0])

    if best_loc is None or best_score < threshold or best_size is None:
        return False

    x, y = best_loc
    w, h = best_size
    page.mouse.click(off_x + x + w // 2, off_y + y + h // 2)
    return True


def _find_left_sidebar(page):
    for sel in [
        "div.sidebar",
        "uni-sidebar",
        "aside",
        "nav",
        "[role='navigation']",
        "div[class*='sidebar']",
    ]:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                box = loc.bounding_box()
                if box and box.get("x", 9999) < 350:
                    return loc
        except Exception:
            pass
    return None


def _click_sidebar_item(page, label: str, timeout: int = 5000) -> bool:
    sidebar = _find_left_sidebar(page)
    if sidebar is None:
        return False

    label_n = normalize(label)

    for sel in ["a", "button", "[role='button']", "[role='link']", "li", "div", "span", "i"]:
        try:
            items = sidebar.locator(sel)
            for i in range(items.count()):
                item = items.nth(i)

                try:
                    text = normalize(item.inner_text(timeout=800))
                except Exception:
                    text = ""

                try:
                    aria = normalize(item.get_attribute("aria-label") or "")
                except Exception:
                    aria = ""

                try:
                    title = normalize(item.get_attribute("title") or "")
                except Exception:
                    title = ""

                hay = " | ".join([text, aria, title])

                if label_n == hay or label_n in hay or text == label_n or aria == label_n or title == label_n:
                    try:
                        item.scroll_into_view_if_needed(timeout=timeout)
                    except Exception:
                        pass

                    try:
                        item.click(timeout=timeout)
                        return True
                    except Exception:
                        try:
                            box = item.bounding_box()
                            if box:
                                page.mouse.click(
                                    box["x"] + box["width"] / 2,
                                    box["y"] + box["height"] / 2,
                                )
                                return True
                        except Exception:
                            pass
        except Exception:
            pass

    try:
        fallback = sidebar.get_by_text(re.compile(rf"^{re.escape(label)}$", re.I)).first
        if fallback.count() > 0:
            fallback.click(timeout=timeout)
            return True
    except Exception:
        pass

    return False


def _modal_visible(page) -> bool:
    try:
        for loc in [
            page.get_by_text(re.compile(r"Bienvenido/?a?", re.I)).first,
            page.locator("[role='dialog']").first,
            page.locator(".modal").first,
            page.locator(".MuiDialog-root").first,
            page.locator(".swal2-popup").first,
            page.locator(".cdk-overlay-pane").first,
        ]:
            try:
                if loc.count() > 0 and loc.is_visible():
                    return True
            except Exception:
                continue
    except Exception:
        pass

    return False


def _dismiss_welcome_modal(page) -> bool:
    deadline = time.time() + 25

    while time.time() < deadline:
        if not _modal_visible(page):
            return True

        _click_first(
            page,
            [
                "[aria-label='Cerrar']",
                "[aria-label='Close']",
                "button[title='Cerrar']",
                "button[title='Close']",
                "button[aria-label*='cerr' i]",
                "button[aria-label*='close' i]",
                "button:has-text('X')",
                "button:has-text('x')",
            ],
            timeout=1000,
        )

        try:
            dialogs = page.locator(
                "[role='dialog'], .modal, .MuiDialog-root, .swal2-popup, .cdk-overlay-pane, .dialog, .popup"
            )
            for i in range(min(dialogs.count(), 6)):
                dialog = dialogs.nth(i)
                try:
                    if not dialog.is_visible():
                        continue
                except Exception:
                    pass

                try:
                    box = dialog.bounding_box()
                except Exception:
                    box = None

                if box and box["width"] > 180 and box["height"] > 100:
                    page.mouse.click(box["x"] + box["width"] - 26, box["y"] + 22)
                    page.wait_for_timeout(500)
        except Exception:
            pass

        if _template_click(page, WELCOME_IMG, threshold=0.80):
            page.wait_for_timeout(500)

        page.wait_for_timeout(400)

        if _modal_visible(page):
            raise RuntimeError("No se pudo cerrar el modal 'Bienvenido/a'.")

    return True


def _set_checkbox_state(page, column_name: str, desired_checked: bool) -> None:
    try:
        row = page.locator("ul.ColVis_collection > li", has_text=column_name).first
        if row.count() > 0:
            checkbox = row.locator("input[type='checkbox']").first
            if checkbox.count() > 0:
                if checkbox.is_checked() != desired_checked:
                    checkbox.click(timeout=3000)
                return
            row.click(timeout=3000)
            return
    except Exception:
        pass

    try:
        checkbox = page.get_by_role("checkbox", name=re.compile(rf"^{re.escape(column_name)}$", re.I)).first
        if checkbox.count() > 0:
            if desired_checked:
                checkbox.check(timeout=3000)
            else:
                checkbox.uncheck(timeout=3000)
            return
    except Exception:
        pass

    try:
        label = page.get_by_text(column_name, exact=True).first
        if label.count() > 0:
            label.click(timeout=3000)
    except Exception:
        pass


def _launch_browser(p):
    for channel in ("msedge", "chrome"):
        try:
            return p.chromium.launch(channel=channel, headless=True)
        except Exception:
            pass
    return p.chromium.launch(headless=True)


def _login(page, username: str, password: str) -> None:
    page.goto(LOGIN_URL, wait_until="domcontentloaded")
    _wait_ready(page)

    try:
        page.get_by_label("Usuario").fill(username)
    except Exception:
        page.locator("input[type='text'], input[name*='user' i], input[id*='user' i]").first.fill(username)

    try:
        page.get_by_label("Password").fill(password)
    except Exception:
        page.locator("input[type='password'], input[name*='pass' i], input[id*='pass' i]").first.fill(password)

    page.wait_for_timeout(1200)

    if not _click_first(
        page,
        [
            "button:has-text('Iniciar Sesión')",
            "button:has-text('Iniciar sesión')",
            "text=Iniciar Sesión",
            "text=Iniciar sesión",
            "[type='submit']",
            "#login button",
            "xpath=//*[@id='login']/form/button",
        ],
    ):
        raise RuntimeError("No encontré el botón de iniciar sesión.")

    _wait_ready(page)


def _goto_default(page) -> None:
    page.goto(DEFAULT_URL, wait_until="domcontentloaded")
    _wait_ready(page)


def _go_to_tracking(page) -> None:
    if _click_sidebar_item(page, "Tracking"):
        _wait_ready(page)
        return

    if _click_first(
        page,
        [
            "div.sidebar div:nth-of-type(3) i",
            "div.sidebar div:nth-of-type(3) a",
            "xpath=//html/body/div[1]/div[1]/div[1]/div[3]/div/a/i",
            "aria/local_shipping",
            "text=Tracking",
        ],
    ):
        _wait_ready(page)
        return

    raise RuntimeError("No encontré 'Tracking'.")


def _go_to_monitoreo(page) -> None:
    _dismiss_welcome_modal(page)

    if _click_first(
        page,
        [
            "uni-sidebar > div > div > div:nth-of-type(3) i",
            "uni-sidebar > div > div > div:nth-of-type(3) a",
            "xpath=//html/body/div[1]/div/div/uni-sidebar/div/div/div[3]/a/i",
            "aria/local_shipping",
        ],
    ):
        _wait_ready(page)
        _dismiss_welcome_modal(page)
        return

    page.goto(TRACKING_HOME_URL, wait_until="domcontentloaded")
    _wait_ready(page)
    _dismiss_welcome_modal(page)


def _open_table_view(page) -> None:
    if _click_first(page, ["#btn-modo-tabla > a", "#btn-modo-tabla", "aria/Tabla", "text/Tabla", "text=Tabla"]):
        _wait_ready(page)
        return
    raise RuntimeError("No encontré la pestaña 'Tabla'.")


def _open_columns_menu(page) -> None:
    if _click_first(
        page,
        [
            "#btn_filtar_col i",
            "#btn_filtar_col",
            "#tablaVehiculo div.columnas_tabla i",
            "xpath=//*[@id='btn_filtar_col']/i",
            "aria/settings",
            "text=settings",
            "text=Seleccionar Columnas",
        ],
        timeout=5000,
    ):
        page.wait_for_timeout(1500)
        return

    region = (
        int(VIEWPORT["width"] * 0.55),
        int(VIEWPORT["height"] * 0.12),
        int(VIEWPORT["width"] * 0.40),
        int(VIEWPORT["height"] * 0.26),
    )

    if _template_click(page, SETTINGS_IMG, region=region, threshold=0.58):
        page.wait_for_timeout(1500)
        return

    raise RuntimeError("No encontré el engranaje de 'Seleccionar Columnas'.")


def _configure_columns(page) -> None:
    _open_columns_menu(page)

    try:
        items = page.locator("ul.ColVis_collection > li")
        for i in range(items.count()):
            item = items.nth(i)
            try:
                text = normalize(item.inner_text(timeout=800))
            except Exception:
                text = ""

            desired = any(normalize(k) == text or normalize(k) in text for k in KEEP_COLUMNS)

            try:
                checkbox = item.locator("input[type='checkbox']").first
                if checkbox.count() > 0:
                    if checkbox.is_checked() != desired:
                        checkbox.click(timeout=2000)
                    page.wait_for_timeout(120)
                    continue
            except Exception:
                pass

            try:
                if desired:
                    item.click(timeout=2000)
            except Exception:
                pass
    except Exception:
        for col in ALL_COLUMNS:
            _set_checkbox_state(page, col, col in KEEP_COLUMNS)

    page.wait_for_timeout(STEP_WAIT_MS)

    if not _click_first(
        page,
        ["div.ColVis_collectionBackground", "xpath=//div[contains(@class,'ColVis_collectionBackground')]"],
        timeout=1500,
    ):
        page.mouse.click(120, 120)

    page.wait_for_timeout(1500)


def _open_export_menu(page) -> None:
    if _click_first(
        page,
        [
            "#btn_exportar span i",
            "#btn_exportar",
            "#tablaVehiculo div.save_tabla i",
            "xpath=//*[@id='btn_exportar']/span/i",
            "aria/launch",
            "text=launch",
            "text=Exportar Tabla",
        ],
        timeout=5000,
    ):
        page.wait_for_timeout(1000)
        return

    region = (
        int(VIEWPORT["width"] * 0.60),
        int(VIEWPORT["height"] * 0.12),
        int(VIEWPORT["width"] * 0.35),
        int(VIEWPORT["height"] * 0.26),
    )

    if _template_click(page, EXPORT_IMG, region=region, threshold=0.58):
        page.wait_for_timeout(1000)
        return

    raise RuntimeError("No encontré el icono de 'Exportar Tabla'.")


def _export_excel(page, download_dir: Path) -> Path:
    with page.expect_download(timeout=25000) as download_info:
        _open_export_menu(page)
        if not _click_first(
            page,
            [
                "button.buttons-excel",
                "aria/Excel",
                "button:has-text('Excel')",
                "button:has-text('EXCEL')",
                "text=Excel",
                "text=EXCEL",
            ],
            timeout=5000,
        ):
            raise RuntimeError("No encontré la opción Excel.")

    download = download_info.value
    download_dir.mkdir(parents=True, exist_ok=True)
    target = download_dir / FIXED_EXPORT_NAME
    download.save_as(target)
    return target


def run_one_cycle(username: str, password: str, download_dir: Path) -> Path:
    with sync_playwright() as p:
        browser = _launch_browser(p)
        context = browser.new_context(accept_downloads=True, viewport=VIEWPORT)
        page = context.new_page()

        try:
            _login(page, username, password)
            _goto_default(page)
            _go_to_tracking(page)
            _go_to_monitoreo(page)
            _open_table_view(page)
            _dismiss_welcome_modal(page)
            _configure_columns(page)
            return _export_excel(page, download_dir)
        finally:
            context.close()
            browser.close()


def start_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText

    verify_runtime_layout()
    cfg = load_config()

    root = tk.Tk()
    root.title("UNIGIS Auto Export")
    root.geometry("1060x760")
    root.minsize(960, 700)
    root.resizable(True, True)

    stop_event = threading.Event()
    run_now_event = threading.Event()
    worker_thread: Optional[threading.Thread] = None

    username_var = tk.StringVar(value=cfg.username)
    password_var = tk.StringVar(value=cfg.password)
    save_var = tk.BooleanVar(value=cfg.save_credentials)
    download_dir_var = tk.StringVar(value=str(_normalize_download_dir(cfg.download_dir)))
    status_var = tk.StringVar(value="Listo.")
    last_update_var = tk.StringVar(value="Última actualización: nunca")
    fixed_path_var = tk.StringVar(value=f"El archivo fijo quedará en: {Path(download_dir_var.get()) / FIXED_EXPORT_NAME}")

    log_box = ScrolledText(root, height=20, wrap="word")
    log_box.configure(state="disabled")

    def refresh_fixed_path_label() -> None:
        current_dir = _normalize_download_dir(download_dir_var.get().strip() or str(DOWNLOAD_DIR_DEFAULT))
        fixed_path_var.set(f"El archivo fijo quedará en: {current_dir / FIXED_EXPORT_NAME}")

    def set_last_update_text(text: str) -> None:
        def _set() -> None:
            last_update_var.set(text)

        root.after(0, _set)

    def log(text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}\n"

        def _append() -> None:
            log_box.configure(state="normal")
            log_box.insert("end", line)
            log_box.see("end")
            log_box.configure(state="disabled")

        status_var.set(text)
        root.update_idletasks()
        root.after(0, _append)

    def set_status(text: str) -> None:
        log(text)

    def choose_download_dir() -> None:
        selected = filedialog.askdirectory(
            title="Elegir carpeta de descarga",
            initialdir=download_dir_var.get() or str(DOWNLOAD_DIR_DEFAULT),
        )
        if selected:
            download_dir_var.set(selected)
            refresh_fixed_path_label()
            set_status(f"Carpeta de descarga: {selected}")

    def ensure_download_dir_if_needed() -> Path:
        candidate = _normalize_download_dir(download_dir_var.get().strip() or str(DOWNLOAD_DIR_DEFAULT))
        if candidate.exists():
            refresh_fixed_path_label()
            return candidate

        root.withdraw()
        try:
            selected = filedialog.askdirectory(
                title="Elige la carpeta donde guardar las descargas",
                initialdir=str(DOWNLOAD_DIR_DEFAULT),
            )
        finally:
            root.deiconify()

        if not selected:
            selected = str(DOWNLOAD_DIR_DEFAULT)

        candidate = _normalize_download_dir(selected)
        candidate.mkdir(parents=True, exist_ok=True)
        download_dir_var.set(str(candidate))
        refresh_fixed_path_label()
        return candidate

    def worker() -> None:
        nonlocal cfg
        try:
            current_user = username_var.get().strip() or cfg.username
            current_pass = password_var.get().strip() or cfg.password
            current_save = bool(save_var.get())
            current_download_dir = _normalize_download_dir(download_dir_var.get().strip() or str(DOWNLOAD_DIR_DEFAULT))
            current_download_dir.mkdir(parents=True, exist_ok=True)

            persist_config(current_user, current_pass, current_save, current_download_dir)
            cfg = AppConfig(current_user, current_pass, current_save, str(current_download_dir))

            while not stop_event.is_set():
                run_now_event.clear()
                set_status("Exportando archivo...")

                try:
                    target = run_one_cycle(current_user, current_pass, current_download_dir)
                    now_text = time.strftime("%d/%m/%Y %H:%M:%S")
                    set_status(f"Descargado: {target.name}")
                    set_last_update_text(f"Última actualización: {now_text}")
                    open_file_default_app(target)
                except PlaywrightTimeoutError as e:
                    set_status(f"Timeout: {e}")
                except Exception as e:
                    set_status(f"Error: {e}")

                if stop_event.is_set():
                    break

                remaining = REFRESH_EVERY_SECONDS
                while remaining > 0 and not stop_event.is_set():
                    if run_now_event.is_set():
                        run_now_event.clear()
                        break

                    time.sleep(1)
                    remaining -= 1

                    if remaining % 60 == 0 and remaining > 0:
                        set_status(f"En espera... próximo ciclo en {remaining // 60} min")

            set_status("Detenido.")
        except Exception as e:
            set_status(f"Error fatal: {e}")

    def on_start() -> None:
        nonlocal worker_thread
        if worker_thread and worker_thread.is_alive():
            messagebox.showinfo("UNIGIS Auto Export", "Ya está corriendo.")
            return

        stop_event.clear()
        current_download_dir = ensure_download_dir_if_needed()
        download_dir_var.set(str(current_download_dir))
        refresh_fixed_path_label()

        persist_config(
            username_var.get().strip(),
            password_var.get().strip(),
            bool(save_var.get()),
            current_download_dir,
        )

        set_status("Iniciando...")
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

    def on_stop() -> None:
        stop_event.set()
        run_now_event.set()
        set_status("Deteniendo...")

    def on_save_and_close() -> None:
        current_download_dir = _normalize_download_dir(download_dir_var.get().strip() or str(DOWNLOAD_DIR_DEFAULT))
        current_download_dir.mkdir(parents=True, exist_ok=True)

        persist_config(
            username_var.get().strip(),
            password_var.get().strip(),
            bool(save_var.get()),
            current_download_dir,
        )
        root.destroy()

    def on_run_now() -> None:
        run_now_event.set()
        set_status("Ejecución solicitada ahora.")

    top = ttk.Frame(root, padding=16)
    top.pack(fill="x")

    ttk.Label(top, text="Usuario").grid(row=0, column=0, sticky="w", pady=(0, 4))
    ttk.Entry(top, textvariable=username_var, width=44).grid(row=0, column=1, sticky="we", pady=(0, 4))

    ttk.Label(top, text="Password").grid(row=1, column=0, sticky="w", pady=(0, 4))
    ttk.Entry(top, textvariable=password_var, width=44, show="*").grid(row=1, column=1, sticky="we", pady=(0, 4))

    ttk.Checkbutton(top, text="Guardar credenciales", variable=save_var).grid(
        row=2, column=1, sticky="w", pady=(6, 10)
    )

    dir_row = ttk.Frame(top)
    dir_row.grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 10))

    ttk.Label(dir_row, text="Carpeta de descarga").pack(side="left")
    ttk.Entry(dir_row, textvariable=download_dir_var, width=60).pack(
        side="left", padx=(10, 8), fill="x", expand=True
    )
    ttk.Button(dir_row, text="Elegir...", command=choose_download_dir).pack(side="left")

    btns = ttk.Frame(top)
    btns.grid(row=4, column=0, columnspan=2, sticky="we", pady=(4, 12))

    ttk.Button(btns, text="Iniciar", command=on_start).pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="Ejecutar ahora", command=on_run_now).pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="Detener", command=on_stop).pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="Guardar y salir", command=on_save_and_close).pack(side="left")

    top.columnconfigure(1, weight=1)

    middle = ttk.Frame(root, padding=(16, 0, 16, 10))
    middle.pack(fill="both", expand=True)

    ttk.Label(middle, text="Registro de ejecución").pack(anchor="w")
    log_box.pack(fill="both", expand=True, pady=(6, 8))

    bottom = ttk.Frame(root, padding=(16, 0, 16, 16))
    bottom.pack(fill="x")

    ttk.Separator(bottom).pack(fill="x", pady=(0, 10))
    ttk.Label(bottom, textvariable=status_var).pack(anchor="w")
    ttk.Label(bottom, textvariable=last_update_var, foreground="#555").pack(anchor="w", pady=(4, 0))
    ttk.Label(bottom, textvariable=fixed_path_var, foreground="#555", wraplength=980).pack(anchor="w", pady=(8, 0))

    def on_close() -> None:
        stop_event.set()
        current_download_dir = _normalize_download_dir(download_dir_var.get().strip() or str(DOWNLOAD_DIR_DEFAULT))
        current_download_dir.mkdir(parents=True, exist_ok=True)

        persist_config(
            username_var.get().strip(),
            password_var.get().strip(),
            bool(save_var.get()),
            current_download_dir,
        )
        root.after(200, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    start_gui()