function activarError(etiqueta){
    const myToast = new bootstrap.Toast(document.querySelector(etiqueta));
    myToast.show();
}