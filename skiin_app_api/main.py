# main.py
import io
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from fastapi import FastAPI, File, HTTPException, UploadFile
from rapidfuzz.fuzz import ratio
from rapidfuzz.process import extractOne
import uvicorn
import pillow_heif

# ------------------ GLOBAL CACHE (Colab'teki reader/df'lerin eşdeğeri) ------------------

model_cache: Dict[str, object] = {}


# ------------------ YARDIMCI FONKSİYONLAR ------------------

def _normalize_turkish(text: str) -> str:
    """Colab'teki _normalize_turkish ile aynı mantık."""
    if not isinstance(text, str):
        return ""
    replacements = {
        "İ": "I", "I": "I", "ı": "I", "i": "I",
        "Ö": "O", "ö": "O",
        "Ü": "U", "ü": "U",
        "Ş": "S", "ş": "S",
        "Ç": "C", "ç": "C",
        "Ğ": "G", "ğ": "G",
    }
    for tr_char, en_char in replacements.items():
        text = text.replace(tr_char, en_char)
    return text.upper()


# --- Colab'teki CHEM_SUFFIXES / TR_STOPWORDS / feature fonksiyonları ---

CHEM_SUFFIXES = [
    "ate", "ite", "ide", "ine", "one", "ol", "yl", "ane", "ene",
    "acid", "oxide", "sulfate", "sulphate", "chloride", "benzoate",
    "glycol", "glycerin", "paraben", "salicylate", "fluoride"
]

TR_STOPWORDS = {
    "kullanım", "kullanın", "kullanılır", "kullanin", "kullanim",
    "çocuk", "cocuk", "çocuklar", "cocuklar",
    "saklayın", "saklayiniz", "saklayıniz",
    "uyarı", "uyarısı", "uyarisi",
    "doktora", "doktorunuza", "doktorunu",
    "yutmayın", "yutmayiniz", "yutulması",
    "için", "icin",
    "günde", "gunde", "kez",
    "ml",
    "yas", "yıl", "yil", "sonra",
    "sicak", "sıcak", "serin",
    "güneş", "gunes", "ışık", "isik",
    "beyaz", "renkli", "çamaşır", "deterjan",
    "etki", "kalıcı",
    "tahris", "goz", "kacinin",
    "responsible", "person", "mame", "name", "and", "eu", "adres",
}

def build_line_features(line: str) -> Dict[str, float]:
    text = line.strip()
    length = len(text)
    tokens = text.split()
    token_count = len(tokens)
    comma_count = text.count(",")
    semi_count = text.count(";")

    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    others = length - letters - digits - spaces

    letter_ratio = letters / length if length > 0 else 0.0
    digit_ratio = digits / length if length > 0 else 0.0
    other_ratio = others / length if length > 0 else 0.0

    chem_hits = 0
    for tok in tokens:
        t = tok.lower().strip(",.;:()[]")
        for suf in CHEM_SUFFIXES:
            if t.endswith(suf):
                chem_hits += 1
                break

    lower = text.lower()
    stop_hits = sum(1 for w in TR_STOPWORDS if w in lower)

    return {
        "length": length,
        "token_count": token_count,
        "comma_count": comma_count,
        "semi_count": semi_count,
        "letter_ratio": letter_ratio,
        "digit_ratio": digit_ratio,
        "other_ratio": other_ratio,
        "chem_hits": chem_hits,
        "stop_hits": stop_hits,
    }


def score_line_as_ingredient(line: str) -> float:
    f = build_line_features(line)
    score = 0.0

    if 1 <= f["token_count"] <= 20:
        score += 1.0
    if f["comma_count"] >= 1:
        score += 1.5
    if f["comma_count"] >= 3:
        score += 0.5

    score += f["chem_hits"] * 1.0

    if f["letter_ratio"] > 0.4:
        score += 0.5
    if f["digit_ratio"] > 0.4 and f["comma_count"] == 0:
        score -= 0.7

    score -= f["stop_hits"] * 1.0

    if f["length"] > 200:
        score -= 1.0

    return score


def _lines_to_ingredient_list(lines: List[str], debug: bool = False) -> List[str]:
    """
    Colab'teki PoC v2/v5'te tanımlanan filtre mantığının aynısı.
    """
    candidates: List[str] = []

    for line in lines:
        line_norm = re.sub(r"\s+", " ", line.strip())
        if not line_norm:
            continue

        if ("," in line_norm) or (";" in line_norm):
            parts = re.split(r"[;,]", line_norm)
        else:
            parts = [line_norm]

        for part in parts:
            ing = part.strip()
            ing = re.sub(r"\s+", " ", ing)
            ing = ing.strip(" .:-|/()[]{}")

            # Çok kısa (örn. 'Se') → at
            if len(ing) < 3:
                if debug:
                    print(f"FİLTRELENDİ (çok kısa): {ing}")
                continue

            # Sadece sayı → at
            if ing.isdigit() or re.fullmatch(r"['\d]+", ing):
                if debug:
                    print(f"FİLTRELENDİ (sadece sayı): {ing}")
                continue

            lower_ing = ing.lower()
            tokens = lower_ing.split()

            # ppm / içerir / florür → genelde açıklama satırı
            if "ppm" in lower_ing or "icerir" in lower_ing or "florur" in lower_ing:
                if debug:
                    print(f"FİLTRELENDİ (ppm/içerik): {ing}")
                continue

            # Çok uzun (7+ kelime) → muhtemelen cümle
            if len(tokens) > 6:
                if debug:
                    print(f"FİLTRELENDİ (çok uzun): {ing}")
                continue

            # Stopword sayısı 2+ ise at
            stop_hits = sum(1 for w in TR_STOPWORDS if w in tokens)
            if stop_hits >= 2:
                if debug:
                    print(f"FİLTRELENDİ (stopword): {ing}")
                continue

            warn_words = [
                "icilmez", "içilmez", "icimez", "içimez",
                "kullaniniz", "kullanin", "kullanın",
            ]
            if any(w in lower_ing for w in warn_words) and len(tokens) <= 4:
                if debug:
                    print(f"FİLTRELENDİ (uyarı): {ing}")
                continue

            # '50 ml', '250ml' vs → at
            tokens_ml = ing.split()
            if len(tokens_ml) <= 3:
                if any(t.isdigit() for t in tokens_ml) and any("ml" in t.lower() for t in tokens_ml):
                    continue

            candidates.append(ing)

    # Tekilleştirme
    seen = set()
    result: List[str] = []
    for ing in candidates:
        key = ing.lower()
        if key not in seen:
            seen.add(key)
            result.append(ing)

    if debug:
        print("---- _lines_to_ingredient_list sonucu (Filtrelenmiş) ----")
        for i, ing in enumerate(result, 1):
            print(f"{i}. {ing}")
        print("-----------------------------------------------------------\n")

    return result


def extract_ingredients(text: str, debug: bool = False) -> List[str]:
    """
    Colab'teki `extract_ingredients` fonksiyonunun aynı mantığı:

    1) Önce 'İÇİNDEKİLER / INGREDIENTS / ISINDEKILER / ...' satırını bul.
       - Bulursa bu satır ve onu takip eden "kimyasal gibi görünen" satırları al.
    2) Hiç keyword bulunamazsa:
       - Satırları score_line_as_ingredient ile puanla,
       - En yüksek skorlu satırın etrafındaki birkaç satırdan liste çıkar.
    """
    raw_lines = [l.rstrip() for l in text.splitlines()]
    lines = [l for l in raw_lines if l.strip()]

    if not lines:
        return []

    keyword_patterns = [
        "INGREDIENTS", "INGREDIEN", "INGREDI",
        "ISINDEKILER", "İÇİNDEKİLER", "ICINDEKILER",
        "ICINDEKI", "ICERIK", "İÇERİK", "CONTENTS",
    ]

    keyword_idx = None
    keyword_line_text = ""
    candidate_lines: List[str] = []

    # 1) Anahtar kelime arama
    for i, line in enumerate(lines):
        up_line = _normalize_turkish(line)

        if keyword_idx is None:
            for pat in keyword_patterns:
                if pat in up_line:
                    keyword_idx = i
                    keyword_line_text = line
                    if debug:
                        print(f"DEBUG: 'İÇİNDEKİLER' anahtar kelimesi {i}. satırda bulundu: {line}")
                    break
        else:
            feats = build_line_features(line)

            # Çok stopword ve virgül yoksa → muhtemelen açıklama, kır
            if feats["stop_hits"] >= 2 and feats["comma_count"] == 0:
                if debug:
                    print(f"DEBUG: Anahtar kelime sonrası {i}. satırda durduruldu (stopword): {line}")
                break

            if (
                feats["comma_count"] > 0
                or feats["chem_hits"] > 0
                or (1 <= feats["token_count"] <= 6)
            ):
                candidate_lines.append(line)

    # Anahtar kelime bulunduysa "aynı satır" kontrolü ve parse
    if keyword_idx is not None:
        if ":" in keyword_line_text:
            same_line_content = keyword_line_text.split(":", 1)[-1]
            if same_line_content.strip():
                candidate_lines.insert(0, same_line_content)
                if debug:
                    print(
                        f"DEBUG: Anahtar kelimeyle aynı satırda içerik bulundu: {same_line_content}"
                    )

        if debug:
            print("Keyword sonrası candidate_lines:")
            for l in candidate_lines:
                print("  -", l)

        ing_list = _lines_to_ingredient_list(candidate_lines, debug=debug)
        if ing_list:
            return ing_list

    # 2) Fallback: satır skorlamaya göre en iyi satırın etrafını al
    scored = []
    for i, line in enumerate(lines):
        s = score_line_as_ingredient(line)
        scored.append((i, line, s))

    scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)

    if not scored_sorted:
        return []

    if debug:
        print("---- Satır skorları (ilk 15) ----")
        for idx, line, s in scored_sorted[:15]:
            print(f"[{idx:02d}] score={s:.2f} | {line}")
        print("---------------------------------")

    best_idx, _, best_score = scored_sorted[0]
    if best_score <= 0:
        return []

    start = max(0, best_idx - 2)
    end = min(len(lines), best_idx + 4)
    candidate_lines_fallback = [lines[k] for k in range(start, end)]

    if debug:
        print("Fallback candidate_lines:")
        for l in candidate_lines_fallback:
            print("  -", l)

    ing_list_fb = _lines_to_ingredient_list(candidate_lines_fallback, debug=debug)
    return ing_list_fb


# --- EŞLEŞTİRME FONKSİYONLARI (Colab'teki v5 mantığı) ---

def find_best_match(noisy_ingredient: str, search_list: List[str], threshold: int = 85):
    if not search_list:
        return noisy_ingredient, 0.0, "Veritabanı Yok"

    term = _normalize_turkish(noisy_ingredient)

    INCI_NAME_LIST = model_cache.get("INCI_NAME_LIST", [])
    TR_TO_INCI_MAP = model_cache.get("TR_TO_INCI_MAP", {})

    # Önce TR→INCI sözlüğüne bak
    if search_list is INCI_NAME_LIST and term in TR_TO_INCI_MAP:
        translated_term = TR_TO_INCI_MAP[term][0]
        if translated_term in INCI_NAME_LIST:
            return translated_term, 100.0, "MATCHED_TRANSLATED"

    match = extractOne(term, search_list, scorer=ratio, score_cutoff=threshold)
    if match:
        best_match, best_score, _ = match
        return best_match, float(best_score), "MATCHED_FUZZY"
    else:
        return term, 0.0, "NO_MATCH"


def find_best_matches_greedy(noisy_part: str, search_list: List[str], threshold: int = 85):
    term_normalized = _normalize_turkish(noisy_part)

    INCI_NAME_LIST = model_cache.get("INCI_NAME_LIST", [])
    TR_TO_INCI_MAP = model_cache.get("TR_TO_INCI_MAP", {})

    if search_list is INCI_NAME_LIST and term_normalized in TR_TO_INCI_MAP:
        translated_list = TR_TO_INCI_MAP[term_normalized]
        matches = []
        for translated_term in translated_list:
            if translated_term in INCI_NAME_LIST:
                matches.append((translated_term, 100.0, "MATCHED_TRANSLATED_SPLIT"))
        if matches:
            return matches

    full_match, full_score, full_status = find_best_match(
        term_normalized, search_list, threshold
    )
    if full_status.startswith("MATCHED"):
        status = (
            "MATCHED_FULL_FUZZY" if full_status == "MATCHED_FUZZY" else full_status
        )
        return [(full_match, full_score, status)]

    tokens = term_normalized.split()
    if not tokens:
        return []

    matches = []
    i = 0
    while i < len(tokens):
        best_token_match = None
        best_token_score = 0.0
        best_token_len = 0
        best_token_status = "NO_MATCH"

        for j in range(i, len(tokens)):
            current_phrase = " ".join(tokens[i : j + 1])
            match, score, status = find_best_match(
                current_phrase, search_list, threshold
            )

            if status.startswith("MATCHED"):
                if score > best_token_score:
                    best_token_match = match
                    best_token_score = score
                    best_token_len = (j - i) + 1
                    best_token_status = status

        if best_token_match:
            matches.append(
                (
                    best_token_match,
                    best_token_score,
                    f"MATCHED_PARTIAL_{best_token_status}",
                )
            )
            i += best_token_len
        else:
            i += 1

    if matches:
        return matches
    else:
        return [(noisy_part, 0.0, "NO_MATCH")]


def find_best_generic_match(noisy_part: str, threshold: int = 85):
    term = _normalize_turkish(noisy_part)
    tokens = term.split()

    GENERIC_NAME_LIST = model_cache.get("GENERIC_NAME_LIST", [])
    matches = []

    for t in tokens:
        match, score, status = find_best_match(t, GENERIC_NAME_LIST, threshold=threshold)
        if status.startswith("MATCHED"):
            matches.append((match, score, "MATCHED_GENERIC"))

    if matches:
        return matches
    else:
        return [(noisy_part, 0.0, "NO_MATCH")]


# ------------------ OCR (Colab'teki ocr_with_easyocr) ------------------

def ocr_with_easyocr(pil_image: Image.Image, debug: bool = False) -> Tuple[str, List[str]]:
    reader = model_cache.get("reader")
    if reader is None:
        raise HTTPException(status_code=500, detail="OCR Modeli yüklenemedi.")

    img = np.array(pil_image)
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img

    result = reader.readtext(img_rgb, detail=1, paragraph=False)
    lines: List[str] = []
    for _, text, _ in result:
        t = text.strip()
        if t:
            lines.append(t)

    if debug:
        print("---- EasyOCR satırları ----")
        for i, l in enumerate(lines, 1):
            print(f"{i:02d}: {l}")
        print("---------------------------")

    full_text = "\n".join(lines)
    return full_text, lines


# ------------------ ANALİZ PIPELINE (Colab'teki ings0 → risk loop) ------------------

def run_analysis_pipeline(pil_image: Image.Image, debug: bool = False) -> Tuple[List[dict], str, List[str]]:
    df_kb = model_cache.get("df_kb")
    df_kb_generic = model_cache.get("df_kb_generic")
    INCI_NAME_LIST = model_cache.get("INCI_NAME_LIST", [])

    if df_kb is None or df_kb_generic is None:
        raise HTTPException(status_code=500, detail="Veritabanları yüklenemedi.")

    # 1) OCR
    text, _ = ocr_with_easyocr(pil_image, debug=debug)

    # 2) İçindekiler listesi (ings0 eşdeğeri)
    noisy_ingredients_list = extract_ingredients(text, debug=debug)

    if debug:
        print("\nÇıkarılan içerik listesi:")
        for i, ing in enumerate(noisy_ingredients_list, 1):
            print(f"{i}. {ing}")
        print("Bulunan ingredient sayısı:", len(noisy_ingredients_list))

    # 3) INCI + Jenerik KB risk analizi (final_ingredients_with_risk)
    final_ingredients_with_risk: List[dict] = []

    for noisy_ing in noisy_ingredients_list:
        ing_to_search = noisy_ing.split(":", 1)[-1].strip()
        ing_to_search = ing_to_search.split(")", 1)[-1].strip()
        ing_to_search = ing_to_search.strip(" .:-|/(){}[]")

        if len(ing_to_search) < 2:
            continue

        matches_inci = find_best_matches_greedy(
            ing_to_search,
            INCI_NAME_LIST,
            threshold=85,
        )

        primary_status = matches_inci[0][2] if matches_inci else "NO_MATCH"

        # 3.1 INCI'de başarılı eşleşmeler
        if primary_status.startswith("MATCHED"):
            for best_match, score, status in matches_inci:
                try:
                    match_row = df_kb[df_kb["INCI_Name"] == best_match].iloc[0]
                    is_restricted = bool(match_row["Is_Restricted"])
                except IndexError:
                    is_restricted = False

                final_ingredients_with_risk.append(
                    {
                        "raw_ocr": noisy_ing,
                        "kb_type": "INCI",
                        "name": best_match,
                        "score": score,
                        "status": status,
                        "is_restricted": is_restricted,
                    }
                )

        # 3.2 INCI'de başarısızsa → Jenerik KB'ye bak
        else:
            term_to_check_generic = matches_inci[0][0] if matches_inci else ing_to_search
            matches_generic = find_best_generic_match(term_to_check_generic, threshold=85)

            for best_match, score, status in matches_generic:
                if status.startswith("MATCHED"):
                    try:
                        match_row = df_kb_generic[
                            df_kb_generic["Generic_Name"] == best_match
                        ].iloc[0]
                        is_restricted = bool(match_row["Is_Restricted"])
                    except IndexError:
                        is_restricted = False

                    final_ingredients_with_risk.append(
                        {
                            "raw_ocr": noisy_ing,
                            "kb_type": "GENERIC",
                            "name": best_match,
                            "score": score,
                            "status": status,
                            "is_restricted": is_restricted,
                        }
                    )

    return final_ingredients_with_risk, text, noisy_ingredients_list


# ------------------ FASTAPI YAŞAM DÖNGÜSÜ (Colab kodunu önceden yükleme) ------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Sunucu başlıyor...")

    # EasyOCR
    print("EasyOCR modeli yükleniyor (gpu=False)...")
    reader = easyocr.Reader(["en", "tr"], gpu=False)
    model_cache["reader"] = reader
    print("EasyOCR yüklendi.")

    # KB 1: INCI
    raw_df = pd.read_csv("COSING_Ingredients-Fragrance Inventory_v2.csv", header=9)
    df_kb = raw_df[[raw_df.columns[1], raw_df.columns[8], raw_df.columns[7]]].copy()
    df_kb.columns = ["INCI_Name", "Function", "COSING_Restriction_Code"]
    df_kb["INCI_Name"] = df_kb["INCI_Name"].str.strip().str.upper()
    df_kb["Is_Restricted"] = df_kb["COSING_Restriction_Code"].notna()
    df_kb.dropna(subset=["INCI_Name"], inplace=True)
    INCI_NAME_LIST = df_kb["INCI_Name"].tolist()

    model_cache["df_kb"] = df_kb
    model_cache["INCI_NAME_LIST"] = INCI_NAME_LIST
    print(f"KB 1 (INCI) yüklendi. {len(INCI_NAME_LIST)} bileşen.")

    # KB 2: Jenerik
    generic_data = {
        "Generic_Name": [
            "NONIYONIK AKTIF MADDE",
            "SABUN",
            "ENZIM",
            "POLIKARBOKSILAT",
            "FOSFONAT",
            "ANYONIK AKTIF MADDE",
            "OPTIK PARLATICI",
            "PARFUM",
            "ESANS",
        ],
        "Function": [
            "Surfactant",
            "Surfactant (Soap)",
            "Cleansing",
            "Polymer",
            "Chelating",
            "Surfactant",
            "Optical Brightener",
            "Fragrance",
            "Fragrance",
        ],
        "Is_Restricted": [False, False, False, False, False, False, False, False, False],
        "Risk_Info": ["", "", "", "", "", "", "", "", ""],
    }
    df_kb_generic = pd.DataFrame(generic_data)
    GENERIC_NAME_LIST = df_kb_generic["Generic_Name"].tolist()

    model_cache["df_kb_generic"] = df_k_generic = df_kb_generic
    model_cache["GENERIC_NAME_LIST"] = GENERIC_NAME_LIST
    print(f"KB 2 (Jenerik) yüklendi. {len(GENERIC_NAME_LIST)} kategori.")

    # TR→INCI çeviri sözlüğü (Colab'teki örnek subset)
    TR_TO_INCI_MAP_RAW = {
        "SORBITOS POLOXAMER 407": ["SORBITOL", "POLOXAMER 407"],
        "SACCHARIN SODIUM FLUORIDE": ["SACCHARIN", "SODIUM FLUORIDE"],
        "ESANS": ["PARFUM"],
        "HİDROKLORİK ASİT": ["HYDROCHLORIC ACID"],
        "YAPAĞI YAĞI": ["LANOLIN"],
        "KOKOAMİD DEA": ["COCAMIDE DEA"],
        "FATTY ALKOL": ["FATTY ALCOHOL"],
        "POLİGLİKOLETER SÜLFOSÜKSİNAT": ["DISODIUM LAURETH SULFOSUCCINATE"],
        "SODYUM LAURİL ETER SÜLFAT": ["SODIUM LAURETH SULFATE"],
        "HİDROJENE GLİSERİL PALMATE": ["HYDROGENATED GLYCERYL PALMATE"],
        "GLİSERİL KOKOAT": ["GLYCERYL COCOATE"],
        "SODYUM HİDROKSİT": ["SODIUM HYDROXIDE"],
        "KUMARİN": ["COUMARIN"],
    }
    model_cache["TR_TO_INCI_MAP"] = {
        _normalize_turkish(k): v for k, v in TR_TO_INCI_MAP_RAW.items()
    }

    print("Sunucu API isteklerini almaya hazır.")
    try:
        yield
    finally:
        model_cache.clear()
        print("Modeller bellekten temizlendi.")


app = FastAPI(lifespan=lifespan)


# ------------------ ENDPOINTLER ------------------

# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Skin App API (Colab PoC mantığıyla) çalışıyor."}


@app.post("/api/scan-image")
async def scan_image_endpoint(image_file: UploadFile = File(...)):

    debug_image_filename = "debug_received_image.jpg" # Kaydedilecek dosya adı

    try:
        image_data = await image_file.read()
        pil_image = Image.open(io.BytesIO(image_data))

        # --- ★★★ KRİTİK DÜZELTME: EXIF Rotasyonu ★★★ ---
        # iOS, fotoğrafı yan kaydedip "EXIF" metadata'sı ile düzeltir.
        # Bu metadata'yı okuyup görseli EasyOCR için düzeltiyoruz.
        pil_image = ImageOps.exif_transpose(pil_image)
        # --- DÜZELTME SONU ---

        # --- Düzeltme 2: RGBA -> RGB'ye Zorlama (Bu kalsın) ---
        if pil_image.mode == 'RGBA' or pil_image.mode == 'P':
            print("DEBUG: Görüntü modu RGBA/P'den RGB'ye dönüştürülüyor...")
            pil_image = pil_image.convert("RGB")

        # --- YENİ ADIM: Görseli Diske Kaydetme ---
        try:
            # Görselin DÖNDÜRÜLMÜŞ halini kaydediyoruz
            pil_image.save(debug_image_filename, "JPEG") 
            print(f"✅ DEBUG: Gelen görsel '{debug_image_filename}' olarak (düzeltilmiş) kaydedildi.")
        except Exception as save_e:
            print(f"❌ DEBUG: Görsel kaydedilemedi: {save_e}")
        # --- KAYDETME SONU ---

    except Exception as e:
        print(f"HATA: Görüntü dosyası okunamadı/dönüştürülemedi: {e}")
        raise HTTPException(status_code=400, detail=f"Geçersiz görüntü dosyası: {e}")

    try:
        # Hata ayıklamayı açık bırakıyoruz ki terminalde görelim
        final_json_output, raw_ocr_text, extracted_candidates = run_analysis_pipeline(
            pil_image, debug=True
        )

        return {
            "analysis_result": final_json_output,
            "debug_raw_ocr": raw_ocr_text,
            "debug_extracted_candidates": extracted_candidates,
        }

    except Exception as e:
        print(f"İşlem sırasında hata: {e}")
        raise HTTPException(
            status_code=500, detail=f"Analiz sırasında sunucu hatası: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
