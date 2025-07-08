
document.addEventListener('DOMContentLoaded', () => {
  const fullBtn     = document.getElementById('btn-full');
  const emptyBtn    = document.getElementById('btn-empty');
  const statusField = document.getElementsByName('status')[0];
  const fileField   = document.getElementById('file');
  const fileList    = document.getElementById('file-list');
  const form        = document.getElementById('upload-form');
  const imagePreview = document.getElementById('image-preview');

  let selectedStatus = 'unknown';

  // Si boutons présents, on les gère
  if (fullBtn && emptyBtn) {
    fullBtn.addEventListener('click', () => {
      selectedStatus = 'dirty';
      statusField.value = selectedStatus;
      fullBtn.classList.add('active');
      emptyBtn.classList.remove('active');
    });

    emptyBtn.addEventListener('click', () => {
      selectedStatus = 'clean';
      statusField.value = selectedStatus;
      emptyBtn.classList.add('active');
      fullBtn.classList.remove('active');
    });
  }

  // Aperçu des fichiers
  fileField.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Affichage de l'image en aperçu
    const reader = new FileReader();
    reader.onload = function(event) {
      imagePreview.src = event.target.result;
      imagePreview.style.display = 'block';  // Affiche l'image
    };
    reader.readAsDataURL(file);

    const dateStr = new Date().toLocaleDateString('fr-FR');

    const emptyRow = fileList.querySelector('tr td[colspan]');
    if (emptyRow) fileList.innerHTML = '';

    const display = selectedStatus === 'dirty' ? 'Pleine'
                  : selectedStatus === 'clean'  ? 'Vide'
                  : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${file.name}</td>
      <td>${dateStr}</td>
      <td>${display}</td>
    `;
    fileList.prepend(tr);
  });


  // Vérification avant envoi
  form.addEventListener('submit', (e) => {
    // Si admin et aucun bouton cliqué
    if ((fullBtn && emptyBtn) && (statusField.value === 'unknown')) {
      e.preventDefault();
      alert('Veuillez sélectionner le statut de la poubelle (pleine ou vide).');
    }
  });
});