import bcrypt
from flask_login import UserMixin
from app import db
from datetime import datetime

class User (db.Model, UserMixin):
    __tablename__ = 'Users'
    id                 = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username           = db.Column(db.String(64), unique=True, nullable=False)
    email              = db.Column(db.String(120), unique=True, nullable=False)
    password           = db.Column(db.String(128), nullable=False)
    nb_pic             = db.Column(db.Integer, default=0)
    role               = db.Column(db.String(200), nullable=False, default='user')
    points             = db.Column(db.Integer, default=0)
    created_at         = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp()) 

    def set_password(self, raw_password: str):
        # bcrypt.gensalt(rounds=5) correspond au deuxième argument de bcrypt.hash() en JS
        salt = bcrypt.gensalt(rounds=5)
        hashed = bcrypt.hashpw(raw_password.encode('utf-8'), salt)
        # stocke la chaîne UTF-8 ($2b$05$…)
        self.password = hashed.decode('utf-8')

    def check_password(self, raw_password: str) -> bool:
        # compare en bytes
        return bcrypt.checkpw(raw_password.encode('utf-8'),
                              self.password.encode('utf-8'))

    @classmethod
    def get_by_email(cls, email):
        """
        Returns the first user matching the given email, or None if not found.
        """
        return cls.query.filter_by(email=email).first()

    @classmethod
    def create(cls, username, email, password):
        """
        Creates a new user, hashes the password, commits to the database, and returns the user instance.
        """
        user = cls(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user

from datetime import datetime
from app import db

class Image(db.Model):
    __tablename__ = 'Images'

    id            = db.Column(
        db.Integer,
        primary_key=True
    )
    file_path     = db.Column(
        db.String(200),
        nullable=False
    )
    label         = db.Column(
        db.String(100),
        nullable=True
    )
    true_label    = db.Column(
        db.String(100),
        nullable=True
    )
    upload_date   = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow
    )
    username      = db.Column(
        db.String(80),
        nullable=False,
        server_default='anonymous'
    )
    localisation  = db.Column(
        db.String(100),
        nullable=False,
        server_default='unknown'
    )
    description   = db.Column(
        db.Text,
        nullable=False,
        server_default='no description'
    )
    quizz         = db.Column(
        db.String(100),
        nullable=True
    )

    def __repr__(self):
        return f'<Image {self.id} – {self.file_path}>'

class FeaturePic(db.Model):
    __tablename__ = 'Features_Pics'

    id        = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(
        db.Text,
        db.ForeignKey('Images.file_path', ondelete='CASCADE'),
        nullable=False
    )
    label     = db.Column(db.Text,   nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    aspect_ratio = db.Column(db.Float, nullable=False)
    file_size_kb = db.Column(db.Float, nullable=False)
    avg_r = db.Column(db.Float, nullable=False)
    var_r = db.Column(db.Float, nullable=False)
    skew_r = db.Column(db.Float, nullable=False)
    avg_g = db.Column(db.Float, nullable=False)
    var_g = db.Column(db.Float, nullable=False)
    skew_g = db.Column(db.Float, nullable=False)
    avg_b = db.Column(db.Float, nullable=False)
    var_b = db.Column(db.Float, nullable=False)
    skew_b = db.Column(db.Float, nullable=False)
    h_hist_0 = db.Column(db.Integer, nullable=False)
    h_hist_1 = db.Column(db.Integer, nullable=False)
    h_hist_2 = db.Column(db.Integer, nullable=False)
    h_hist_3 = db.Column(db.Integer, nullable=False)
    h_hist_4 = db.Column(db.Integer, nullable=False)
    h_hist_5 = db.Column(db.Integer, nullable=False)
    h_hist_6 = db.Column(db.Integer, nullable=False)
    h_hist_7 = db.Column(db.Integer, nullable=False)
    h_hist_8 = db.Column(db.Integer, nullable=False)
    h_hist_9 = db.Column(db.Integer, nullable=False)
    h_hist_10 = db.Column(db.Integer, nullable=False)
    h_hist_11 = db.Column(db.Integer, nullable=False)
    h_hist_12 = db.Column(db.Integer, nullable=False)
    h_hist_13 = db.Column(db.Integer, nullable=False)
    h_hist_14 = db.Column(db.Integer, nullable=False)
    h_hist_15 = db.Column(db.Integer, nullable=False)
    h_hist_16 = db.Column(db.Integer, nullable=False)
    h_hist_17 = db.Column(db.Integer, nullable=False)
    h_hist_18 = db.Column(db.Integer, nullable=False)
    h_hist_19 = db.Column(db.Integer, nullable=False)
    s_hist_0 = db.Column(db.Integer, nullable=False)
    s_hist_1 = db.Column(db.Integer, nullable=False)
    s_hist_2 = db.Column(db.Integer, nullable=False)
    s_hist_3 = db.Column(db.Integer, nullable=False)
    s_hist_4 = db.Column(db.Integer, nullable=False)
    s_hist_5 = db.Column(db.Integer, nullable=False)
    s_hist_6 = db.Column(db.Integer, nullable=False)
    s_hist_7 = db.Column(db.Integer, nullable=False)
    s_hist_8 = db.Column(db.Integer, nullable=False)
    s_hist_9 = db.Column(db.Integer, nullable=False)
    s_hist_10 = db.Column(db.Integer, nullable=False)
    s_hist_11 = db.Column(db.Integer, nullable=False)
    s_hist_12 = db.Column(db.Integer, nullable=False)
    s_hist_13 = db.Column(db.Integer, nullable=False)
    s_hist_14 = db.Column(db.Integer, nullable=False)
    s_hist_15 = db.Column(db.Integer, nullable=False)
    s_hist_16 = db.Column(db.Integer, nullable=False)
    s_hist_17 = db.Column(db.Integer, nullable=False)
    s_hist_18 = db.Column(db.Integer, nullable=False)
    s_hist_19 = db.Column(db.Integer, nullable=False)
    v_hist_0 = db.Column(db.Integer, nullable=False)
    v_hist_1 = db.Column(db.Integer, nullable=False)
    v_hist_2 = db.Column(db.Integer, nullable=False)
    v_hist_3 = db.Column(db.Integer, nullable=False)
    v_hist_4 = db.Column(db.Integer, nullable=False)
    v_hist_5 = db.Column(db.Integer, nullable=False)
    v_hist_6 = db.Column(db.Integer, nullable=False)
    v_hist_7 = db.Column(db.Integer, nullable=False)
    v_hist_8 = db.Column(db.Integer, nullable=False)
    v_hist_9 = db.Column(db.Integer, nullable=False)
    v_hist_10 = db.Column(db.Integer, nullable=False)
    v_hist_11 = db.Column(db.Integer, nullable=False)
    v_hist_12 = db.Column(db.Integer, nullable=False)
    v_hist_13 = db.Column(db.Integer, nullable=False)
    v_hist_14 = db.Column(db.Integer, nullable=False)
    v_hist_15 = db.Column(db.Integer, nullable=False)
    v_hist_16 = db.Column(db.Integer, nullable=False)
    v_hist_17 = db.Column(db.Integer, nullable=False)
    v_hist_18 = db.Column(db.Integer, nullable=False)
    v_hist_19 = db.Column(db.Integer, nullable=False)
    gray_hist_0 = db.Column(db.Integer, nullable=False)
    gray_hist_1 = db.Column(db.Integer, nullable=False)
    gray_hist_2 = db.Column(db.Integer, nullable=False)
    gray_hist_3 = db.Column(db.Integer, nullable=False)
    gray_hist_4 = db.Column(db.Integer, nullable=False)
    gray_hist_5 = db.Column(db.Integer, nullable=False)
    gray_hist_6 = db.Column(db.Integer, nullable=False)
    gray_hist_7 = db.Column(db.Integer, nullable=False)
    gray_hist_8 = db.Column(db.Integer, nullable=False)
    gray_hist_9 = db.Column(db.Integer, nullable=False)
    gray_hist_10 = db.Column(db.Integer, nullable=False)
    gray_hist_11 = db.Column(db.Integer, nullable=False)
    gray_hist_12 = db.Column(db.Integer, nullable=False)
    gray_hist_13 = db.Column(db.Integer, nullable=False)
    gray_hist_14 = db.Column(db.Integer, nullable=False)
    gray_hist_15 = db.Column(db.Integer, nullable=False)
    gray_hist_16 = db.Column(db.Integer, nullable=False)
    gray_hist_17 = db.Column(db.Integer, nullable=False)
    gray_hist_18 = db.Column(db.Integer, nullable=False)
    gray_hist_19 = db.Column(db.Integer, nullable=False)
    lum_hist_0 = db.Column(db.Integer, nullable=False)
    lum_hist_1 = db.Column(db.Integer, nullable=False)
    lum_hist_2 = db.Column(db.Integer, nullable=False)
    lum_hist_3 = db.Column(db.Integer, nullable=False)
    lum_hist_4 = db.Column(db.Integer, nullable=False)
    lum_hist_5 = db.Column(db.Integer, nullable=False)
    lum_hist_6 = db.Column(db.Integer, nullable=False)
    lum_hist_7 = db.Column(db.Integer, nullable=False)
    lum_hist_8 = db.Column(db.Integer, nullable=False)
    lum_hist_9 = db.Column(db.Integer, nullable=False)
    lum_hist_10 = db.Column(db.Integer, nullable=False)
    lum_hist_11 = db.Column(db.Integer, nullable=False)
    lum_hist_12 = db.Column(db.Integer, nullable=False)
    lum_hist_13 = db.Column(db.Integer, nullable=False)
    lum_hist_14 = db.Column(db.Integer, nullable=False)
    lum_hist_15 = db.Column(db.Integer, nullable=False)
    lum_hist_16 = db.Column(db.Integer, nullable=False)
    lum_hist_17 = db.Column(db.Integer, nullable=False)
    lum_hist_18 = db.Column(db.Integer, nullable=False)
    lum_hist_19 = db.Column(db.Integer, nullable=False)
    contrast = db.Column(db.Integer, nullable=False)
    laplacian_var = db.Column(db.Float, nullable=False)
    canny_count = db.Column(db.Integer, nullable=False)
    sobel_count = db.Column(db.Integer, nullable=False)
    edge_density = db.Column(db.Float, nullable=False)
    center_edge = db.Column(db.Integer, nullable=False)
    surround_edge = db.Column(db.Integer, nullable=False)
    hog_0 = db.Column(db.Float, nullable=False)
    hog_1 = db.Column(db.Float, nullable=False)
    hog_2 = db.Column(db.Float, nullable=False)
    hog_3 = db.Column(db.Float, nullable=False)
    hog_4 = db.Column(db.Float, nullable=False)
    hog_5 = db.Column(db.Float, nullable=False)
    hog_6 = db.Column(db.Float, nullable=False)
    hog_7 = db.Column(db.Float, nullable=False)
    hog_8 = db.Column(db.Float, nullable=False)
    hog_9 = db.Column(db.Float, nullable=False)
    hog_10 = db.Column(db.Float, nullable=False)
    hog_11 = db.Column(db.Float, nullable=False)
    hog_12 = db.Column(db.Float, nullable=False)
    hog_13 = db.Column(db.Float, nullable=False)
    hog_14 = db.Column(db.Float, nullable=False)
    hog_15 = db.Column(db.Float, nullable=False)
    hog_16 = db.Column(db.Float, nullable=False)
    hog_17 = db.Column(db.Float, nullable=False)
    hog_18 = db.Column(db.Float, nullable=False)
    hog_19 = db.Column(db.Float, nullable=False)
    hog_20 = db.Column(db.Float, nullable=False)
    hog_21 = db.Column(db.Float, nullable=False)
    hog_22 = db.Column(db.Float, nullable=False)
    hog_23 = db.Column(db.Float, nullable=False)
    hog_24 = db.Column(db.Float, nullable=False)
    hog_25 = db.Column(db.Float, nullable=False)
    hog_26 = db.Column(db.Float, nullable=False)
    hog_27 = db.Column(db.Float, nullable=False)
    hog_28 = db.Column(db.Float, nullable=False)
    hog_29 = db.Column(db.Float, nullable=False)
    hog_30 = db.Column(db.Float, nullable=False)
    hog_31 = db.Column(db.Float, nullable=False)
    hog_32 = db.Column(db.Float, nullable=False)
    hog_33 = db.Column(db.Float, nullable=False)
    hog_34 = db.Column(db.Float, nullable=False)
    hog_35 = db.Column(db.Float, nullable=False)
    hog_36 = db.Column(db.Float, nullable=False)
    hog_37 = db.Column(db.Float, nullable=False)
    hog_38 = db.Column(db.Float, nullable=False)
    hog_39 = db.Column(db.Float, nullable=False)
    hog_40 = db.Column(db.Float, nullable=False)
    hog_41 = db.Column(db.Float, nullable=False)
    hog_42 = db.Column(db.Float, nullable=False)
    hog_43 = db.Column(db.Float, nullable=False)
    hog_44 = db.Column(db.Float, nullable=False)
    hog_45 = db.Column(db.Float, nullable=False)
    hog_46 = db.Column(db.Float, nullable=False)
    hog_47 = db.Column(db.Float, nullable=False)
    hog_48 = db.Column(db.Float, nullable=False)
    hog_49 = db.Column(db.Float, nullable=False)
    lbp_0 = db.Column(db.Float, nullable=False)
    lbp_1 = db.Column(db.Float, nullable=False)
    lbp_2 = db.Column(db.Float, nullable=False)
    lbp_3 = db.Column(db.Float, nullable=False)
    lbp_4 = db.Column(db.Float, nullable=False)
    lbp_5 = db.Column(db.Float, nullable=False)
    lbp_6 = db.Column(db.Float, nullable=False)
    lbp_7 = db.Column(db.Float, nullable=False)
    lbp_8 = db.Column(db.Float, nullable=False)
    lbp_9 = db.Column(db.Float, nullable=False)
    glcm_contrast = db.Column(db.Float, nullable=False)
    glcm_dissimilarity = db.Column(db.Float, nullable=False)
    glcm_homogeneity = db.Column(db.Float, nullable=False)
    glcm_energy = db.Column(db.Float, nullable=False)
    glcm_correlation = db.Column(db.Float, nullable=False)
    glcm_ASM = db.Column(db.Float, nullable=False)
    fft_energy = db.Column(db.Float, nullable=False)
    orb_keypoints = db.Column(db.Integer, nullable=False)
    blob_count = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        """
        Convertit une instance de FeaturePic en dictionnaire.
        """
        return {
            'id': self.id,
            'file_path': self.file_path,
            'label': self.label,
            'width': self.width,
            'height': self.height,
            'aspect_ratio': self.aspect_ratio,
            'file_size_kb': self.file_size_kb,
            'avg_r': self.avg_r,
            'var_r': self.var_r,
            'skew_r': self.skew_r,
            'avg_g': self.avg_g,
            'var_g': self.var_g,
            'skew_g': self.skew_g,
            'avg_b': self.avg_b,
            'var_b': self.var_b,
            'skew_b': self.skew_b,
            'h_hist_0': self.h_hist_0,
            'h_hist_1': self.h_hist_1,
            'h_hist_2': self.h_hist_2,
            'h_hist_3': self.h_hist_3,
            'h_hist_4': self.h_hist_4,
            'h_hist_5': self.h_hist_5,
            'h_hist_6': self.h_hist_6,
            'h_hist_7': self.h_hist_7,
            'h_hist_8': self.h_hist_8,
            'h_hist_9': self.h_hist_9,
            'h_hist_10': self.h_hist_10,
            'h_hist_11': self.h_hist_11,
            'h_hist_12': self.h_hist_12,
            'h_hist_13': self.h_hist_13,
            'h_hist_14': self.h_hist_14,
            'h_hist_15': self.h_hist_15,
            'h_hist_16': self.h_hist_16,
            'h_hist_17': self.h_hist_17,
            'h_hist_18': self.h_hist_18,
            'h_hist_19': self.h_hist_19,
            's_hist_0': self.s_hist_0,
            's_hist_1': self.s_hist_1,
            's_hist_2': self.s_hist_2,
            's_hist_3': self.s_hist_3,
            's_hist_4': self.s_hist_4,
            's_hist_5': self.s_hist_5,
            's_hist_6': self.s_hist_6,
            's_hist_7': self.s_hist_7,
            's_hist_8': self.s_hist_8,
            's_hist_9': self.s_hist_9,
            's_hist_10': self.s_hist_10,
            's_hist_11': self.s_hist_11,
            's_hist_12': self.s_hist_12,
            's_hist_13': self.s_hist_13,
            's_hist_14': self.s_hist_14,
            's_hist_15': self.s_hist_15,
            's_hist_16': self.s_hist_16,
            's_hist_17': self.s_hist_17,
            's_hist_18': self.s_hist_18,
            's_hist_19': self.s_hist_19,
            'v_hist_0': self.v_hist_0,
            'v_hist_1': self.v_hist_1,
            'v_hist_2': self.v_hist_2,
            'v_hist_3': self.v_hist_3,
            'v_hist_4': self.v_hist_4,
            'v_hist_5': self.v_hist_5,
            'v_hist_6': self.v_hist_6,
            'v_hist_7': self.v_hist_7,
            'v_hist_8': self.v_hist_8,
            'v_hist_9': self.v_hist_9,
            'v_hist_10': self.v_hist_10,
            'v_hist_11': self.v_hist_11,
            'v_hist_12': self.v_hist_12,
            'v_hist_13': self.v_hist_13,
            'v_hist_14': self.v_hist_14,
            'v_hist_15': self.v_hist_15,
            'v_hist_16': self.v_hist_16,
            'v_hist_17': self.v_hist_17,
            'v_hist_18': self.v_hist_18,
            'v_hist_19': self.v_hist_19,
            'gray_hist_0': self.gray_hist_0,
            'gray_hist_1': self.gray_hist_1,
            'gray_hist_2': self.gray_hist_2,
            'gray_hist_3': self.gray_hist_3,
            'gray_hist_4': self.gray_hist_4,
            'gray_hist_5': self.gray_hist_5,
            'gray_hist_6': self.gray_hist_6,
            'gray_hist_7': self.gray_hist_7,
            'gray_hist_8': self.gray_hist_8,
            'gray_hist_9': self.gray_hist_9,
            'gray_hist_10': self.gray_hist_10,
            'gray_hist_11': self.gray_hist_11,
            'gray_hist_12': self.gray_hist_12,
            'gray_hist_13': self.gray_hist_13,
            'gray_hist_14': self.gray_hist_14,
            'gray_hist_15': self.gray_hist_15,
            'gray_hist_16': self.gray_hist_16,
            'gray_hist_17': self.gray_hist_17,
            'gray_hist_18': self.gray_hist_18,
            'gray_hist_19': self.gray_hist_19,
            'lum_hist_0': self.lum_hist_0,
            'lum_hist_1': self.lum_hist_1,
            'lum_hist_2': self.lum_hist_2,
            'lum_hist_3': self.lum_hist_3,
            'lum_hist_4': self.lum_hist_4,
            'lum_hist_5': self.lum_hist_5,
            'lum_hist_6': self.lum_hist_6,
            'lum_hist_7': self.lum_hist_7,
            'lum_hist_8': self.lum_hist_8,
            'lum_hist_9': self.lum_hist_9,
            'lum_hist_10': self.lum_hist_10,
            'lum_hist_11': self.lum_hist_11,
            'lum_hist_12': self.lum_hist_12,
            'lum_hist_13': self.lum_hist_13,
            'lum_hist_14': self.lum_hist_14,
            'lum_hist_15': self.lum_hist_15,
            'lum_hist_16': self.lum_hist_16,
            'lum_hist_17': self.lum_hist_17,
            'lum_hist_18': self.lum_hist_18,
            'lum_hist_19': self.lum_hist_19,
            'contrast': self.contrast,
            'laplacian_var': self.laplacian_var,
            'canny_count': self.canny_count,
            'sobel_count': self.sobel_count,
            'edge_density': self.edge_density,
            'center_edge': self.center_edge,
            'surround_edge': self.surround_edge,
            'hog_0': self.hog_0,
            'hog_1': self.hog_1,
            'hog_2': self.hog_2,
            'hog_3': self.hog_3,
            'hog_4': self.hog_4,
            'hog_5': self.hog_5,
            'hog_6': self.hog_6,
            'hog_7': self.hog_7,
            'hog_8': self.hog_8,
            'hog_9': self.hog_9,
            'hog_10': self.hog_10,
            'hog_11': self.hog_11,
            'hog_12': self.hog_12,
            'hog_13': self.hog_13,
            'hog_14': self.hog_14,
            'hog_15': self.hog_15,
            'hog_16': self.hog_16,
            'hog_17': self.hog_17,
            'hog_18': self.hog_18,
            'hog_19': self.hog_19,
            'hog_20': self.hog_20,
            'hog_21': self.hog_21,
            'hog_22': self.hog_22,
            'hog_23': self.hog_23,
            'hog_24': self.hog_24,
            'hog_25': self.hog_25,
            'hog_26': self.hog_26,
            'hog_27': self.hog_27,
            'hog_28': self.hog_28,
            'hog_29': self.hog_29,
            'hog_30': self.hog_30,
            'hog_31': self.hog_31,
            'hog_32': self.hog_32,
            'hog_33': self.hog_33,
            'hog_34': self.hog_34,
            'hog_35': self.hog_35,
            'hog_36': self.hog_36,
            'hog_37': self.hog_37,
            'hog_38': self.hog_38,
            'hog_39': self.hog_39,            
            'hog_40': self.hog_40,
            'hog_41': self.hog_41,
            'hog_42': self.hog_42,
            'hog_43': self.hog_43,
            'hog_44': self.hog_44,
            'hog_45': self.hog_45,
            'hog_46': self.hog_46,
            'hog_47': self.hog_47,
            'hog_48': self.hog_48,
            'hog_49': self.hog_49,
            'lbp_0': self.lbp_0,
            'lbp_1': self.lbp_1,
            'lbp_2': self.lbp_2,
            'lbp_3': self.lbp_3,
            'lbp_4': self.lbp_4,
            'lbp_5': self.lbp_5,
            'lbp_6': self.lbp_6,
            'lbp_7': self.lbp_7,
            'lbp_8': self.lbp_8,
            'lbp_9': self.lbp_9,
            'glcm_contrast': self.glcm_contrast,
            'glcm_dissimilarity': self.glcm_dissimilarity,
            'glcm_homogeneity': self.glcm_homogeneity,
            'glcm_energy': self.glcm_energy,
            'glcm_correlation': self.glcm_correlation,
            'glcm_ASM': self.glcm_ASM,
            'fft_energy': self.fft_energy,
            'orb_keypoints': self.orb_keypoints,
            'blob_count': self.blob_count
        }
    
    
class ClassificationRule(db.Model):
    __tablename__ = 'classification_rules'

    id = db.Column(db.Integer, primary_key=True)
    mode = db.Column(db.String(20), default='auto')
    file_size_threshold = db.Column(db.Integer, default=2500)
    avg_r_threshold = db.Column(db.Integer, default=127)
    avg_g_threshold = db.Column(db.Integer, default=127)
    avg_b_threshold = db.Column(db.Integer, default=127)
    contrast_threshold = db.Column(db.Float, default=200.0)
    laplacian_var_threshold = db.Column(db.Float, default=10000.0)

    def __repr__(self):
        return f"<ClassificationRule {self.id}>"