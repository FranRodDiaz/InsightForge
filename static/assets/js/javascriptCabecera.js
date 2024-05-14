

    // When the user clicks on the button, scroll to the top of the document
    function topFunction() {
      document.body.scrollTop = 0; // For Safari
      document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
    }

        function adjustNav() {
        var width = window.innerWidth;
        var nav = document.getElementById('my-nav');

        if (width >= 992) { // 992px es el breakpoint para 'lg' en Bootstrap 5
            nav.classList.add('nav', 'nav-tabs');
        } else {
            nav.classList.remove('nav', 'nav-tabs');
        }
    }

    // Ejecutar al cargar y al cambiar el tamaño de la ventana
    window.onload = adjustNav;
    window.onresize = adjustNav;