from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, BooleanField, SubmitField, HiddenField, IntegerField, FloatField
from wtforms.validators import DataRequired, EqualTo, Email

class LoginForm(FlaskForm):
    email        = StringField('Email', validators=[DataRequired(), Email()])
    password     = PasswordField('Mot de passe', validators=[DataRequired()])
    remember     = BooleanField('Se souvenir de moi')
    submit_login = SubmitField('Se connecter')

class RegisterForm(FlaskForm):
    username        = StringField('Nom d’utilisateur', validators=[DataRequired()])
    email           = StringField('Email', validators=[DataRequired(), Email()])
    password        = PasswordField(
        'Mot de passe',
        validators=[DataRequired(), EqualTo('confirm', message='Les mots de passe doivent correspondre')]
    )
    confirm         = PasswordField('Confirmez le mot de passe')
    submit_register = SubmitField('S’inscrire')

class ChangePasswordForm(FlaskForm):
    old_password = PasswordField(
        'Ancien mot de passe',
        validators=[DataRequired(message="Veuillez renseigner votre mot de passe actuel.")]
    )
    new_password = PasswordField(
        'Nouveau mot de passe',
        validators=[
            DataRequired(message="Le nouveau mot de passe ne peut pas être vide."),
            EqualTo('confirm', message='Les mots de passe doivent correspondre.')
        ]
    )
    confirm = PasswordField(
        'Confirmez le mot de passe',
        validators=[DataRequired(message="Veuillez confirmer votre nouveau mot de passe.")]
    )
    submit_pwd = SubmitField('Changer le mot de passe')


class UploadAvatarForm(FlaskForm):
    avatar = FileField(
        'Nouvel avatar',
        validators=[
            FileRequired(message="Vous devez sélectionner une image."),
            FileAllowed(['jpg', 'jpeg', 'png', 'gif', 'webp'], 'Seules les images sont autorisées !')
        ],
        render_kw={'accept': 'image/jpeg,image/png,image/gif,image/webp'}
    )
    submit_avatar = SubmitField('Uploader')  # ← ajoutez cette ligne

class UploadForm(FlaskForm):
    file = FileField(
        'Sélectionner un fichier',
        validators=[
            FileRequired(message="Vous devez choisir un fichier."),
            FileAllowed(['jpg','jpeg','png','gif','webp', 'heic'], 'Images seulement !')
        ],
        render_kw={
            'accept': 'image/jpeg,image/png,image/gif,image/webp,image/heic'
        }
    )
    status = HiddenField('Statut', default='unknown')
    submit = SubmitField('Téléverser et classifier')
    
class ClassificationRuleForm(FlaskForm):
    file_size_threshold = IntegerField(
        'Taille du fichier (Ko)', 
        default=2000,
        validators=[DataRequired()],
        render_kw={"min": 100, "max": 5000, "step": 50}  # Min, max et step pour le curseur
    )
    avg_r_threshold = IntegerField(
        'Rouge moyen (avg_r)', 
        default=127,
        validators=[DataRequired()],
        render_kw={"min": 0, "max": 255, "step": 1}  # Min, max et step pour le curseur
    )
    avg_g_threshold = IntegerField(
        'Vert moyen (avg_g)', 
        default=127,
        validators=[DataRequired()],
        render_kw={"min": 0, "max": 255, "step": 1}
    )
    avg_b_threshold = IntegerField(
        'Bleu moyen (avg_b)', 
        default=127,
        validators=[DataRequired()],
        render_kw={"min": 0, "max": 255, "step": 1}
    )
    contrast_threshold = FloatField(
        'Contraste', 
        default=230.0,
        validators=[DataRequired()],
        render_kw={"min": 110, "max": 255, "step": 1}  # Min, max et step pour le curseur
    )
    laplacian_var_threshold = FloatField(
        'Variance Laplacienne', 
        default=3000.0,
        validators=[DataRequired()],
        render_kw={"min": 100, "max": 20000, "step": 200}  # Min, max et step pour le curseur
    )
    submit = SubmitField('Mettre à jour les règles')
    reset = SubmitField('Réinitialiser par défaut (Mode auto)')
