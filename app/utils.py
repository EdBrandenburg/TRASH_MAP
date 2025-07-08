import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
import random
import uuid
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
from sqlalchemy.orm import Session
from flask import current_app
from app import db
from app.models import FeaturePic, Image, ClassificationRule
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

def extract_exif_data(img_path):
    """Retourne (datetime, localisation) extraits depuis les EXIF."""
    try:
        image = PILImage.open(img_path)
        exif_data = image._getexif()
        if not exif_data:
            return None, None

        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

        # 1) Date
        date_str = exif.get('DateTimeOriginal') or exif.get('DateTime')
        if date_str:
            try:
                dt = datetime.datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                dt = None
        else:
            dt = None

        # 2) GPS
        gps_info_raw = exif.get("GPSInfo")
        if gps_info_raw:
            gps_info = {GPSTAGS.get(t, t): gps_info_raw[t] for t in gps_info_raw}

            lat = gps_info.get('GPSLatitude')
            lat_ref = gps_info.get('GPSLatitudeRef')
            lon = gps_info.get('GPSLongitude')
            lon_ref = gps_info.get('GPSLongitudeRef')

            if lat and lat_ref and lon and lon_ref:
                lat_deg = _convert_to_degrees(lat, lat_ref)
                lon_deg = _convert_to_degrees(lon, lon_ref)
                if lat_deg is not None and lon_deg is not None:
                    loc = f"{lat_deg:.6f},{lon_deg:.6f}"
                else:
                    loc = None
            else:
                loc = None
        else:
            loc = None

        return dt, loc

    except Exception as e:
        print(f"[EXIF ERROR] {e}")
        return None, None


def _convert_to_degrees(value, ref):
    try:
        def rational_to_float(r):
            return float(r.num) / float(r.den) if hasattr(r, 'num') else float(r)

        d = rational_to_float(value[0])
        m = rational_to_float(value[1])
        s = rational_to_float(value[2])

        coord = d + (m / 60.0) + (s / 3600.0)
        if ref in ['S', 'W']:
            coord = -coord
        return coord
    except Exception as e:
        print(f"[GPS CONVERT ERROR] {e}")
        return None

def generate_unique_filename(filename):
    # Générer un UUID unique pour garantir l'unicité du fichier
    unique_id = str(uuid.uuid4())  # UUID quasi aucune chance doublons
    # Extraire l'extension du fichier
    file_extension = filename.rsplit('.', 1)[-1]
    return f"{unique_id}.{file_extension}"

def compress_image(img_path, max_size=(800, 800), quality=80):
    try:
        with PILImage.open(img_path) as img:
            # Redimensionner si besoin
            img.thumbnail(max_size)
            # Construire nouveau nom de fichier en WebP
            webp_path = img_path.rsplit('.', 1)[0] + '.webp'
            # Sauvegarder en WebP
            img.save(webp_path, format='WEBP', quality=quality)

        # Supprimer l'image originale (non-webp)
        if img_path != webp_path:
            os.remove(img_path)

        return webp_path

    except Exception as e:
        print(f"[ERREUR compress_image] : {e}")
        return None
  


# Paramètres globaux
LBP_RADIUS       = 1
LBP_N_POINTS     = 8 * LBP_RADIUS
LBP_METHOD       = 'uniform'
LBP_N_BINS       = LBP_N_POINTS + 2

HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS    = 9

GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS     = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']

def extract_features_dict(img_path: str, label: str='unknown') -> dict | None:
    """Extrait les features d'une image et renvoie un dict, ou None si erreur."""
    try:
        # PIL pour dims & taille fichier
        pil = PILImage.open(img_path)
        w, h = pil.size
        area = w * h
        
        # OpenCV pour traitements
        img_cv      = cv2.imread(img_path)
        img_resized = cv2.resize(img_cv, (128,128))
        gray        = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hsv         = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        arr128      = np.array(pil.resize((128,128)))
        
        feat = {
            'width'        : w,
            'height'       : h,
            'aspect_ratio' : round(w/h,3),
            'file_size_kb' : round(os.path.getsize(img_path)/1024.0,2),
            'label'        : label
        }
        
        # Moyenne / variance / skew par canal RGB
        for idx,ch in enumerate(('r','g','b')):
            vals = arr128[:,:,idx].ravel().astype(np.float32)
            feat[f'avg_{ch}']  = float(vals.mean())
            feat[f'var_{ch}']  = float(vals.var())
            feat[f'skew_{ch}'] = float(((vals-vals.mean())**3).mean() / (vals.std()**3 + 1e-6))
        
        # Hist H/S/V (20 bins)
        for i,chan in enumerate((gray,hsv[:,:,1],hsv[:,:,2])):
            hist = cv2.calcHist([chan],[0],None,[256],[0,256]).flatten()[:20]
            key = ('h','s','v')[i]
            for j,v in enumerate(hist):
                feat[f'{key}_hist_{j}'] = int(v)
        
        # Hist gray & luminaire
        ghist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()[:20]
        for i,v in enumerate(ghist): feat[f'gray_hist_{i}'] = int(v)
        lum   = (0.299*arr128[:,:,0] + 0.587*arr128[:,:,1] + 0.114*arr128[:,:,2]).astype(np.uint8)
        lhist = cv2.calcHist([lum],[0],None,[256],[0,256]).flatten()[:20]
        for i,v in enumerate(lhist): feat[f'lum_hist_{i}'] = int(v)
        
        # Contraste, Laplacian variance
        feat['contrast']     = int(gray.max()-gray.min())
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        feat['laplacian_var']= float(lap.var())
        
        # Canny / Sobel / densité
        canny_img = cv2.Canny(gray, 100, 200) > 0
        sobx    = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        soby    = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        sobel_img = (np.sqrt(sobx**2 + soby**2) > 50)
        
        feat.update({
            'canny_count' : int(canny_img.sum()),
            'sobel_count' : int(sobel_img.sum()),
            'edge_density': float(canny_img.sum() / (area + 1e-6))
        })
        
        # Centre vs périphérie
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        half = h // 4
        mask = np.zeros_like(gray, dtype=bool)
        mask[cy-half:cy+half, cx-half:cx+half] = True

        feat['center_edge']   = int(canny_img[mask].sum())
        feat['surround_edge'] = int(canny_img[~mask].sum())
        
        # HOG 50 premiers
        hogf = hog(gray,
                   orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm='L2-Hys',
                   feature_vector=True)[:50]
        for i,v in enumerate(hogf): feat[f'hog_{i}'] = float(v)
        
        # LBP 20 premiers
        lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
        lh,_= np.histogram(lbp.ravel(), bins=LBP_N_BINS, range=(0,LBP_N_BINS))
        lh = lh.astype(float)/(lh.sum()+1e-6)
        for i,v in enumerate(lh[:20]): feat[f'lbp_{i}'] = float(v)
        
        # GLCM
        glcm = graycomatrix(gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                            symmetric=True, normed=True)
        for prop in GLCM_PROPS:
            feat[f'glcm_{prop}'] = float(graycoprops(glcm, prop).mean())
        
        # FFT energy
        f = np.fft.fft2(gray); fshift=np.fft.fftshift(f)
        feat['fft_energy'] = float(np.log1p(np.abs(fshift)).sum())
        
        # ORB keypoints & blobs
        orb = cv2.ORB_create(); kp = orb.detect(gray, None)
        feat['orb_keypoints'] = len(kp)
        blobs = cv2.SimpleBlobDetector_create().detect(gray)
        feat['blob_count']    = len(blobs)
        
        return feat

    except Exception as e:
        current_app.logger.error(f"Feature extraction failed for {img_path}: {e}")
        return None


def extract_and_save(img_disk_path: str, relative_path: str, label: str='unknown') -> FeaturePic | None:
    """
    Extrait et stocke dans la BDD un FeaturePic.
     - img_disk_path: chemin absolu sur disque (pour OpenCV/PIL).
     - relative_path: chemin stocké en base (correspond à Features_Pics.file_path FK Images.file_path).
    """
    data = extract_features_dict(img_disk_path, label)
    if not data:
        return None

    # Créer l'instance FeaturePic
    pic = FeaturePic(
        file_path      = relative_path,
        label          = data.pop('label'),
        width          = data.pop('width'),
        height         = data.pop('height'),
        aspect_ratio   = data.pop('aspect_ratio'),
        file_size_kb   = data.pop('file_size_kb'),
        **data
    )
    db.session.add(pic)
    db.session.commit()
    return pic


### Classification
class RuleBasedClassifier:
    def __init__(self, feature_stats: dict, top_n: int = 20, delta: float = 0.5):
        self.feature_stats = feature_stats
        self.top_n = top_n
        self.delta = delta
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['score'], reverse=True)
        self.top_features = [feat for feat, _ in sorted_features[:top_n]]

    def classify(self, features: dict) -> str:
        total = 0.0
        print(f"Top features: {self.top_features}")
        for feat in self.top_features:
            stat = self.feature_stats[feat]
            t = stat['threshold']
            std = stat['std']
            weight = stat['score']
            z = (features[feat] - t) / std if std != 0 else 0
            if stat['direction'] == 'dirty':
                z = -z
            total += weight * z
        return 'clean' if total > self.delta else 'dirty'
    

def classify_image(db_session: Session, features_dict: dict, img_id: int) -> str:
    """
    Classifie une image en utilisant soit un classifieur automatique (z-score),
    soit les règles manuelles, selon le mode défini.
    """

    # Charger le mode de classification et les seuils
    rule = db_session.query(ClassificationRule).first()
    mode = rule.mode if rule else 'auto'

    if mode == 'manual':
        # === MODE MANUEL ===
        try:
            fs = features_dict

            file_size_ok = fs.get('file_size', 0) > rule.file_size_threshold
            contrast_ok = fs.get('contrast', 0) > rule.contrast_threshold
            lap_ok = fs.get('laplacian_var', 0) < rule.laplacian_var_threshold
            avg_r_ok = fs.get('avg_r', 0) > rule.avg_r_threshold
            avg_g_ok = fs.get('avg_g', 0) > rule.avg_g_threshold
            avg_b_ok = fs.get('avg_b', 0) > rule.avg_b_threshold

            if file_size_ok and contrast_ok and not lap_ok:
                return 'dirty'
            elif not file_size_ok and avg_r_ok and avg_g_ok and avg_b_ok:
                return 'clean'
            else:
                return 'unknown'
        except Exception as e:
            print(f"[Erreur Classification - Mode manuel] {e}")
            return 'unknown'

    else:
        # === MODE AUTO ===
        feats = db_session.query(FeaturePic).all()
        feature_dicts = [f.to_dict() for f in feats]
        df_features = pd.DataFrame(feature_dicts)

        images = db_session.query(Image).all()
        image_dicts = [{'file_path': img.file_path, 'true_label': img.true_label} for img in images]
        df_labels = pd.DataFrame(image_dicts)

        df = pd.merge(df_features, df_labels, on='file_path', how='left')
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        if 'true_label' not in df.columns or df['true_label'].dropna().empty:
            return 'unknown'

        numeric_cols = get_numeric_features(df)
        feature_stats = compute_feature_stats(df, numeric_cols)

        classifier = RuleBasedClassifier(feature_stats=feature_stats, top_n=20, delta=0.5)
        predicted_label = classifier.classify(features_dict)

        print(f"Image ID {img_id} classified as {predicted_label} (mode auto).")
        return predicted_label


def get_numeric_features(df: pd.DataFrame, exclude: list = ['width', 'height']) -> list:
    """
    Retrieve numeric columns excluding those that are not useful.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    return numeric_cols

def compute_feature_stats(df: pd.DataFrame, numeric_cols: list, label_col: str = 'true_label') -> dict:
    """
    For each numeric feature, compute the discriminative power as |Δmean|/std over the first 40 rows.
    
    The threshold is defined as the average of the means for the 'clean' and 'dirty' groups.
    The 'direction' indicates which group has the higher mean.
    """
    stats = {}
    labeled = df.loc[:39]  # use only the labeled subset
    for col in numeric_cols:
        clean_vals = labeled[labeled[label_col] == 'clean'][col]
        dirty_vals = labeled[labeled[label_col] == 'dirty'][col]
        std = labeled[col].std(ddof=0)
        if std == 0:
            continue
        d_mean = abs(clean_vals.mean() - dirty_vals.mean())
        score = d_mean / std
        stats[col] = {
            'score': score,
            'threshold': (clean_vals.mean() + dirty_vals.mean()) / 2,
            'std': std,
            'direction': 'clean' if clean_vals.mean() > dirty_vals.mean() else 'dirty'
        }
    return stats

def get_random_location():
    """Retourne une coordonnée aléatoire autour de Paris, sous forme de string 'lat, lon'"""
    lat = round(random.uniform(48.82, 48.90), 6)  # Paris : latitude
    lon = round(random.uniform(2.28, 2.42), 6)    # Paris : longitude
    return f"{lat}, {lon}"


import sqlite3

# Chemin vers la base de données (adapte-le si besoin)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '../Database', 'database_factory.db')

def connect_db():
    """Connexion à la base de données"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  
    return conn

def get_total_images(conn):
    """Retourne le nombre total d'images"""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS total FROM Images")
    return cursor.fetchone()["total"]

def get_images_by_user(conn):
    """Retourne un dictionnaire : {username: nb_photos}"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT username, COUNT(*) AS total 
        FROM Images 
        GROUP BY username
        ORDER BY total DESC
    """)
    results = cursor.fetchall()
    return [
        {"username": row["username"], "total": row["total"]}
        for row in results
    ]

def get_trash_status_counts(conn):
    """Retourne le nombre de poubelles 'plein' et 'vide' selon label"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT label, COUNT(*) AS count
        FROM Images
        WHERE label IN ('dirty', 'clean')
        GROUP BY label
    """)
    results = cursor.fetchall()
    counts = {"dirty": 0, "clean": 0}
    for row in results:
        counts[row["label"]] = row["count"]
    return counts

def get_label_accuracy(conn):
    """
    Compare 'label' et 'true_label' et retourne un dict avec :
    - false_clean (predicted clean but actually dirty)
    - true_clean (predicted clean and actually clean)
    - false_dirty (predicted dirty but actually clean)
    - true_dirty (predicted dirty and actually dirty)
    """
    cursor = conn.cursor()
    query = """
        SELECT 
            SUM(CASE WHEN label = 'clean' AND true_label = 'dirty' THEN 1 ELSE 0 END) AS false_clean,
            SUM(CASE WHEN label = 'clean' AND true_label = 'clean' THEN 1 ELSE 0 END) AS true_clean,
            SUM(CASE WHEN label = 'dirty' AND true_label = 'clean' THEN 1 ELSE 0 END) AS false_dirty,
            SUM(CASE WHEN label = 'dirty' AND true_label = 'dirty' THEN 1 ELSE 0 END) AS true_dirty
        FROM Images
    """
    cursor.execute(query)
    row = cursor.fetchone()
    return {
        "false_clean": row["false_clean"],
        "true_clean": row["true_clean"],
        "false_dirty": row["false_dirty"],
        "true_dirty": row["true_dirty"]
    }

import matplotlib
matplotlib.use('Agg')

def generate_label_pie_chart(data):
    labels = [
        "Faux vide",
        "Vrai vide",
        "Faux plein",
        "Vrai plein"
    ]
    sizes = [
        data["false_clean"],
        data["true_clean"],
        data["false_dirty"],
        data["true_dirty"]
    ]
    colors = ['#ff6384', '#36a2eb', '#ffcd56', '#4bc0c0']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal') 

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def get_arrondissement_from_coords(lat, lon):
    """
    Retourne l'arrondissement de Paris (1 à 20), 'Villejuif', ou un code département (92, 93, 94, 91)
    en fonction des coordonnées GPS.
    Renvoie None si les coordonnées ne correspondent à aucune zone définie.
    """
    arr_zones = {
        1:  (48.859, 48.865, 2.326, 2.342),
        2:  (48.865, 48.871, 2.33, 2.345),
        3:  (48.862, 48.87, 2.355, 2.37),
        4:  (48.852, 48.86, 2.35, 2.365),
        5:  (48.837, 48.85, 2.34, 2.355),
        6:  (48.845, 48.855, 2.325, 2.34),
        7:  (48.85, 48.865, 2.305, 2.325),
        8:  (48.87, 48.88, 2.31, 2.33),
        9:  (48.875, 48.885, 2.33, 2.355),
        10: (48.875, 48.89, 2.355, 2.375),
        11: (48.855, 48.87, 2.37, 2.395),
        12: (48.835, 48.85, 2.39, 2.42),
        13: (48.815, 48.835, 2.355, 2.38),
        14: (48.825, 48.84, 2.305, 2.34),
        15: (48.83, 48.855, 2.28, 2.31),
        16: (48.85, 48.88, 2.25, 2.29),
        17: (48.885, 48.9, 2.295, 2.32),
        18: (48.885, 48.9, 2.34, 2.37),
        19: (48.88, 48.9, 2.375, 2.4),
        20: (48.86, 48.875, 2.39, 2.42),
    }

    for arr, (lat_min, lat_max, lon_min, lon_max) in arr_zones.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return f"{arr}e arrondissement"

    # Villejuif 
    if 48.78 <= lat <= 48.8 and 2.34 <= lon <= 2.37:
        return "Villejuif"

    # Hauts-de-Seine (92) 
    if 48.8 <= lat <= 48.9 and 2.18 <= lon <= 2.3:
        return "92"

    # Seine-Saint-Denis (93) 
    if 48.88 <= lat <= 48.95 and 2.4 <= lon <= 2.5:
        return "93"

    # Val-de-Marne (94) 
    if 48.75 <= lat <= 48.82 and 2.4 <= lon <= 2.52:
        return "94"

    # Essonne (91) 
    if 48.5 <= lat <= 48.75 and 2.2 <= lon <= 2.45:
        return "91"

    return "Hors des limites"  



import ast

def get_labels_by_arrondissement(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT localisation, label FROM Images WHERE label IN ('clean', 'dirty')")
    rows = cursor.fetchall()

    result = {}

    for row in rows:
        try:
            localisation = ast.literal_eval(row["localisation"])
            lat, lon = float(localisation[0]), float(localisation[1])
            arr = get_arrondissement_from_coords(lat, lon)
            if arr is None:
                continue

            if arr not in result:
                result[arr] = {"clean": 0, "dirty": 0}

            result[arr][row["label"]] += 1

        except Exception:
            continue  

    return {
        f"{arr}e arrondissement" if isinstance(arr, int) else arr: values
        for arr, values in sorted(result.items())
    }


def generate_pie_for_arrondissement(data, arrondissement):
    """
    Génère un camembert pour un arrondissement donné.
    `data` est le dictionnaire retourné par get_labels_by_arrondissement
    """
    labels = ["Poubelles pleines", "Poubelles vides"]
    colors = ['#ff6384', '#36a2eb']

    arr_data = data.get(arrondissement)
    if not arr_data:
        return None  # Aucun point dans cet arrondissement

    values = [arr_data["dirty"], arr_data["clean"]]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title(f"Répartition {arrondissement}")
    ax.axis('equal')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded

def get_dashboard_data():
    conn = connect_db()

    labels_by_arr = get_labels_by_arrondissement(conn)

    data = {
        "total_images": get_total_images(conn),
        "images_by_user": get_images_by_user(conn),
        "trash_status_counts": get_trash_status_counts(conn),
        "label_accuracy": get_label_accuracy(conn),
        "labels_by_arrondissement": labels_by_arr,
        "pie_16e": generate_pie_for_arrondissement(labels_by_arr, "16e arrondissement"),
        "pie_15e": generate_pie_for_arrondissement(labels_by_arr, "15e arrondissement")
    }

    conn.close()
    return data


def get_quizz_images(conn):
    """Retourne 2 images connues (avec true_label) + 1 image inconnue"""
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, file_path, true_label 
        FROM Images 
        WHERE true_label IN ('clean', 'dirty')
        ORDER BY RANDOM()
        LIMIT 2
    """)
    known_images = cursor.fetchall()

    cursor.execute("""
        SELECT id, file_path 
        FROM Images 
        WHERE true_label = 'unknown'
        ORDER BY RANDOM()
        LIMIT 1
    """)
    unknown_image = cursor.fetchone()

    return known_images, unknown_image

def validate_quizz_responses(known_ids, known_guesses, unknown_id, unknown_guess):
    conn = connect_db()
    cursor = conn.cursor()

    # Vérifier les réponses des images connues
    cursor.execute(
        "SELECT id, true_label FROM Images WHERE id IN (?, ?)",
        (known_ids[0], known_ids[1])
    )
    known_images = cursor.fetchall()

    correct = True
    for i, row in enumerate(known_images):
        if known_guesses[i] != row["true_label"]:
            correct = False
            break

    if not correct:
        conn.close()
        return

    # Récupérer la valeur actuelle du champ Quizz
    cursor.execute("SELECT Quizz FROM Images WHERE id = ?", (unknown_id,))
    row = cursor.fetchone()
    current_quizz = row["Quizz"] if row else None
    
    if current_quizz == "pachyderm":
        # Cas 1 : on remplace Quizz par la réponse de l'utilisateur
        cursor.execute(
            "UPDATE Images SET Quizz = ? WHERE id = ?",
            (unknown_guess, unknown_id)
        )
    else:
        if unknown_guess == current_quizz:
            # Cas 2a : réponse correcte → on écrit dans true_label
            cursor.execute(
                "UPDATE Images SET true_label = ? WHERE id = ?",
                (unknown_guess, unknown_id)
            )
        else:
            # Cas 2b : réponse incorrecte → on met "pachyderm"
            cursor.execute(
                "UPDATE Images SET Quizz = ? WHERE id = ?",
                ("pachyderm", unknown_id)
            )

    conn.commit()
    conn.close()

